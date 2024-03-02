import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
import cupy as cp
from PIL import Image
from scipy.signal import savgol_filter
import math
import time
import warnings

import json
import os

import umap

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import warnings
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from cffi.backend_ctypes import xrange
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""## Load Data"""


# Kmeans--init hook definition


@torch.no_grad()
def data_dependent_init_forward_hook(
    self, inputs, outputs, use_kmeans=True, verbose=False
):
    """initializes codebook from data"""

    if (not self.training) or (self.data_initialized.item() == 1):
        return

    if verbose:
        print("initializing codebook with k-means++")

    def sample_centroids(z_e, num_codes):
        """replaces the data of the codebook with z_e randomly."""

        z_e = z_e.reshape(-1, z_e.size(-1))

        if num_codes >= z_e.size(0):
            e_msg = (
                f"\ncodebook size > warmup samples: {num_codes} vs {z_e.size(0)}. "
                + "recommended to decrease the codebook size or increase batch size."
            )

            # warnings.warn(str(cs(e_msg, 'yellow')))

            # repeat until it fits and add noise
            repeat = num_codes // z_e.shape[0]
            new_codes = z_e.data.tile([repeat, 1])[:num_codes]
            new_codes += 1e-3 * torch.randn_like(new_codes.data)

        else:
            # you have more warmup samples than codebook. subsample data
            if use_kmeans:
                from torchpq.clustering import KMeans

                kmeans = KMeans(
                    n_clusters=num_codes, distance="euclidean", init_mode="kmeans++"
                )
                kmeans.fit(z_e.data.T.contiguous())
                new_codes = kmeans.centroids.T
            else:
                indices = torch.randint(low=0, high=num_codes, size=(num_codes,))
                indices = indices.to(z_e.device)
                new_codes = torch.index_select(z_e, 0, indices).to(z_e.device).data

        return new_codes

    _, z_q, _, _, z_e = outputs
    # the z_e here is the output of encoder(i.e. the input of vqvae) which has been reshaped to [b,h,w,c]

    if type(self) is VectorQuantizerEMA:
        num_codes = self._embedding.weight.shape[0]
        new_codebook = sample_centroids(z_e, num_codes)
        self._embedding.weight.data = new_codebook

    self.data_initialized.fill_(1)
    return


class SoftClustering:
    def __init__(self, delta=0.2, lr=0.0):
        """
        Initialize the SoftClustering class.

        Args:
            delta (float): The update rate (a constant).
        """
        self.delta = delta
        self.epsilon = torch.tensor(0.5)
        self.lr = lr
        self.sim_matrix = None

    def _compute_similarity(self, v1, v2):
        distances = torch.norm(v1 - v2, p=2, dim=2)
        similarity = torch.exp(1.0 / (distances + self.epsilon))
        return similarity

    def update_delta(self):
        self.delta -= self.delta * self.lr

    # @torch.no_grad()
    # def update_sim_matrix(self, codebook):
    #     """
    #     Perform soft clustering update on signal points.
    #
    #     Args:
    #         codebook (torch.Tensor): The original codebook.
    #
    #     Returns:
    #         newcodebook (torch.Tensor): The updated codebook.
    #     """
    #
    #     num_codes, feature_dim = codebook.shape
    #
    #     codebook_expanded = codebook.unsqueeze(0).expand(num_codes, -1, -1)
    #     codebook_transposed = codebook_expanded.permute(1, 0, 2)
    #     distances = torch.norm(codebook_expanded - codebook_transposed, p=2, dim=2)
    #     self.sim_matrix = torch.exp(1.0 / (distances + self.epsilon))
    #
    #     del distances, codebook_expanded, codebook_transposed
    #     # return codebook
    @torch.no_grad()
    def update_sim_matrix(self, codebook):
        """
        Perform similarity matrix update based on current codebook.

        Args:
            codebook (torch.Tensor): The current codebook.

        Results:
            update sim_matrix (torch.Tensor)
        """

        num_codes, feature_dim = codebook.shape
        # compute distances and similarity
        codebook_expanded = codebook.unsqueeze(0).expand(num_codes, -1, -1)
        # shape: [num_codes, num_codes, embedding_dim]

        # transpose the 1st and 2nd dim of codebook_expanded
        codebook_transposed = codebook_expanded.permute(1, 0, 2)
        # compute distances along the 3rd dim with l2 norm, forming a distance matrix of shape [num_codes, num_codes]
        # in which, distances[i, j] indicate the distance between the ith and jth code vectors
        distances = torch.norm(codebook_expanded - codebook_transposed, p=2, dim=2)
        # compute similarity, the larger the distance, the smaller the similarity
        self.sim_matrix = torch.exp(-distances)

    def compute_weighted_sum(self, codebook):
        """
        Compute the second term of equation 11, i.e. compute the weighted sum of other codes for any code

        Args:
            codebook (torch.Tensor): The current codebook.

        Return:
            weighted_codebook_sum (torch.Tensor)
        """
        # set the diagonal elements as 0, as we don't want to the code itself to be part of weighted sum
        self.sim_matrix.fill_diagonal_(0)
        # compute sum of similarity of each row
        sim_sum = torch.sum(self.sim_matrix, dim=1)
        # expand dim of sim_sum
        sim_sum = sim_sum.unsqueeze(1)
        # normalize the similarity
        weights = self.sim_matrix / sim_sum
        # multiply other codes by their normalized similarity
        weighted_codebook_sum = torch.matmul(weights, codebook)
        return weighted_codebook_sum


# test_cluster = SoftClustering()
# import time
# import torch

# _num = 2048
# _dim = 64
# test_input = torch.rand([_num, _dim]).to(device)
# test_cluster.update_codebook(test_input)

"""## Vector Quantizer Layer

This layer takes a tensor to be quantized. The channel dimension will be used as the space in which to quantize. All other dimensions will be flattened and will be seen as different examples to quantize.

The output tensor will have the same shape as the input.

As an example for a `BCHW` tensor of shape `[16, 64, 32, 32]`, we will first convert it to an `BHWC` tensor of shape `[16, 32, 32, 64]` and then reshape it into `[16384, 64]` and all `16384` vectors of size `64`  will be quantized independently. In otherwords, the channels are used as the space in which to quantize. All other dimensions will be flattened and be seen as different examples to quantize, `16384` in this case.
"""


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost,
        kmeans_init=True,
        soft_discretization=False,
        gamma=0.8,
        gamma_lr=1 / 10000,
        soft_clustering=False,
        delta=0.2,
        delta_lr=0,
        every_cluster_iters=250,
        delta_decrease_threshold=0,
    ):

        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        if kmeans_init:
            self.register_buffer("data_initialized", torch.zeros(1))
            self.register_forward_hook(data_dependent_init_forward_hook)

        self.soft_discretization = soft_discretization
        self.gamma = gamma
        self.gamma_lr = gamma_lr

        self.delta = delta
        self.delta_lr = delta_lr
        self.every_cluster_iters = every_cluster_iters
        self.decrease_threshold = delta_decrease_threshold

        if soft_clustering:
            self.soft_cluster_assignment = SoftClustering(
                delta=self.delta, lr=self.delta_lr
            )

        self.training_iter_num = 0

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)  # [BHW, C]

        # perform codebook update
        if self.training:
            if hasattr(self, "soft_cluster_assignment"):
                if (
                    self.soft_cluster_assignment.delta > 0
                    and self.training_iter_num % self.every_cluster_iters == 0
                ):  # 每迭代完一次所有batch更新一下codebook
                    self.soft_cluster_assignment.update_sim_matrix(
                        self._embedding.weight
                    )
                if self.soft_cluster_assignment.delta > 0:
                    self.soft_cluster_assignment.update_codebook(self._embedding.weight)
                if (
                    self.soft_cluster_assignment.lr > 0
                    and self.training_iter_num >= self.decrease_threshold
                ):
                    self.soft_cluster_assignment.update_delta()
            self.training_iter_num += 1

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )  # [BHW, num_embeddings] 对于任一个连续向量，储存其和所有code的距离

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # [BHW, 1]
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )  # [BHW, num_embeddings]
        encodings.scatter_(
            1, encoding_indices, 1
        )  # [BHW, num_embeddings] 对于任一个连续向量，以one-hot的形式储存和其距离最小的code的指标

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(
            input_shape
        )  # [B, H, W, embedding_dim] 对于任一个连续向量，储存和其距离最小的code

        # Soft discretization
        if self.training:
            if self.soft_discretization:
                if self.gamma > 0:
                    quantized = (1 - self.gamma) * quantized + self.gamma * inputs
                    self.gamma -= self.gamma * self.gamma_lr

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        c_loss = F.mse_loss(quantized, inputs.detach())
        loss = c_loss + self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + torch.tensor(1e-10)))
        )

        # convert quantized from BHWC -> BCHW
        return (
            loss,
            quantized.permute(0, 3, 1, 2).contiguous(),
            perplexity,
            encodings,
            inputs,
        )


"""We will also implement a slightly modified version  which will use exponential moving averages to update the 
embedding vectors instead of an auxillary loss. This has the advantage that the embedding updates are independent of 
the choice of optimizer for the encoder, decoder and other parts of the architecture. For most experiments the EMA 
version trains faster than the non-EMA version."""


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost,
        decay,
        epsilon=1e-5,
        kmeans_init=True,
        soft_discretization=False,
        gamma=0.2,
        gamma_lr=1 / 10000,
        soft_clustering=False,
        delta=0.2,
        delta_lr=1 / 10000,
        every_cluster_iters=250,
        delta_decrease_threshold=0,
    ):

        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer(
            "_ema_cluster_size", torch.zeros(num_embeddings)
        )  # 用来跟踪每个码字的累计分配样本数，buffer不参与训练
        self._ema_w = nn.Parameter(
            torch.Tensor(num_embeddings, self._embedding_dim)
        )  # [_num, _dim] 是模型参数，表示在指数移动平均中使用的权重向量，参与训练
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

        if kmeans_init:
            self.register_buffer("data_initialized", torch.zeros(1))
            self.register_forward_hook(data_dependent_init_forward_hook)

        self.soft_discretization = soft_discretization
        self.gamma = gamma
        self.gamma_lr = gamma_lr

        self.delta = delta
        self.delta_lr = delta_lr
        self.every_cluster_iters = every_cluster_iters
        self.decrease_threshold = delta_decrease_threshold

        if soft_clustering:
            self.sca = SoftClustering(delta=self.delta, lr=self.delta_lr)

        self.training_iter_num = 0

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(
            -1, self._embedding_dim
        )  # [num_signals, C] num_signals = B*H*W

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )  # [num_signals, num_embeddings] store distances

        # Encoding find the indices for nearest neighbor
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(
            1
        )  # [num_signals, 1]
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )  # [num_signals, num_embeddings]
        encodings.scatter_(1, encoding_indices, 1)
        # [num_signals, num_embeddings] one-hot indices for nearest neighbors

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(
            input_shape
        )  # i.e. z_q, shape: [B, H, W, embedding_dim]

        # Soft discretization and Soft cluster assignment
        if self.training:
            if self.soft_discretization and hasattr(self, "soft_cluster_assignment"):
                if self.gamma > 0:
                    quantized = (1 - self.gamma) * quantized + self.gamma * inputs
                    self.gamma -= self.gamma * self.gamma_lr
                if self.sca.delta > 0:
                    self.sca.update_sim_matrix(self._embedding.weight)
                    weighted_sum = self.sca.compute_weighted_sum(self._embedding.weight)
                    # compute the second term of equ 11
                    sca_term = torch.matmul(encodings, weighted_sum).view(input_shape)
                    # perform SCA
                    quantized = (
                        1 - self.sca.delta
                    ) * quantized + self.sca.delta * sca_term
                    # update delta
                    self.sca.update_delta()
            elif self.soft_discretization:
                if self.gamma > 0:
                    quantized = (1 - self.gamma) * quantized + self.gamma * inputs
                    self.gamma -= self.gamma * self.gamma_lr
            elif hasattr(self, "soft_cluster_assignment"):
                if self.sca.delta > 0:
                    self.sca.update_sim_matrix(self._embedding.weight)
                    weighted_sum = self.sca.compute_weighted_sum(self._embedding.weight)
                    # compute the second term of equ 11
                    sca_term = torch.matmul(encodings, weighted_sum).view(input_shape)
                    # perform SCA
                    quantized = (
                        1 - self.sca.delta
                    ) * quantized + self.sca.delta * sca_term
                    # update delta
                    self.sca.update_delta()
            self.training_iter_num += 1
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(
                encodings, 0
            )  # （num_embedding, ), track the
            # assigment numbers

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )  # 对 _ema_cluster_size
            # 进行Laplace平滑，从而避免分配样本数为零的问题，同时也在计算码字的平均值时引入一些平滑调整

            dw = torch.matmul(
                encodings.t(), flat_input
            )  # (BHW, num).t .* (BHW, dim) = (num, dim),
            # 这一步是对每一个聚类加权平均，dw每一行代表一个聚类，行向量代表属于该聚类的flat_input中的sample加和的结果
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw
            )  # EMA更新codebook，第二个term
            # 使得codebook向encoder representation的方向更新

            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
            )  # 进行加权平均

        # Loss
        # Here we use quantized.detach() to make loss has no relationship with codebook, so codebook would not be
        # updated during bp
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        c_latent_loss = F.mse_loss(inputs.detach(), quantized)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + torch.tensor(1e-10)))
        )

        # convert quantized from BHWC -> BCHW
        return (
            loss,
            quantized.permute(0, 3, 1, 2).contiguous(),
            perplexity,
            encodings,
            inputs,
        )
