import argparse
import pickle

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import lmdb
from tqdm import tqdm

from dataset import ImageFileDataset, CodeRow
from vqvae import VQVAE


def extract(lmdb_env, loader, model, device):
    index = 0

    with lmdb_env.begin(write=True) as txn:
        pbar = tqdm(loader)

        for img, _, filename in pbar:
            img = img.to(device)

            _, _, _, id_t, id_b = model.encode(img)
            id_t = id_t.detach().cpu().numpy()
            id_b = id_b.detach().cpu().numpy()

            for file, top, bottom in zip(filename, id_t, id_b):
                row = CodeRow(top=top, bottom=bottom, filename=file)
                txn.put(str(index).encode("utf-8"), pickle.dumps(row))
                index += 1
                pbar.set_description(f"inserted: {index}")

        txn.put("length".encode("utf-8"), str(index).encode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("path", type=str)

    args = parser.parse_args()

    device = "cuda"

    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = ImageFileDataset(args.path, transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)

    vq_kwargs = {
        "soft_discretization": True,
        "gamma": 0.2,
        "gamma_lr": 0.0002,
        "soft_clustering": True,
        "delta": 0.1,
        "delta_lr": 0.0002,
        "kmeans_init": False,
    }

    model = VQVAE(channel=32, n_embed=128, embed_dim=16, **vq_kwargs)
    model.load_state_dict(torch.load(args.ckpt))
    model = model.to(device)
    model.eval()

    map_size = 100 * 1024 * 1024 * 1024

    env = lmdb.open(args.name, map_size=map_size)

    extract(env, loader, model, device)
