import os
import torch
import random
import json
import numpy as np
from glob import glob
from tqdm import tqdm
from numba import jit
from torch.utils.data import DataLoader
from torchvision.datasets.folder import DatasetFolder

EXT = (".ply",)


def read_ply(path):

    file = open(path, "r")
    safe_number = 20
    for i in range(safe_number):
        line = file.readline().strip()
        if "element vertex" in line:
            n_verts = int(line.split(" ")[2])

        elif "element face" in line:
            n_faces = int(line.split(" ")[2])

        elif "end_header" in line:
            break

    verts = [
        [float(s) for s in file.readline().strip().split(" ")[0:3]] for i_vert in range(n_verts)
    ]
    faces = [
        [int(s) for s in file.readline().strip().split(" ")[1:4]] for i_face in range(n_faces)
    ]
    file.close()
    return torch.tensor(verts), torch.tensor(faces), path


def read_off(path):

    file = open(path, "r")
    off_header = file.readline().strip()
    if "OFF" == off_header:
        n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(" ")])
    else:
        n_verts, n_faces, __ = tuple([int(s) for s in off_header[3:].split(" ")])

    verts = [[float(s) for s in file.readline().strip().split(" ")] for i_vert in range(n_verts)]
    faces = [
        [int(s) for s in file.readline().strip().split(" ")][1:] for i_face in range(n_faces)
    ]

    file.close()
    return torch.tensor(verts), torch.tensor(faces)


@jit(nopython=True)
def triangle_area(pt1, pt2, pt3):
    side_a = np.linalg.norm(pt1 - pt2)
    side_b = np.linalg.norm(pt2 - pt3)
    side_c = np.linalg.norm(pt3 - pt1)
    # Heron's Formula for the triangle area.
    u = 0.5 * (side_a + side_b + side_c)
    return max(u * (u - side_a) * (u - side_b) * (u - side_c), 0) ** 0.5


@jit(nopython=True)
def sample_point(pt1, pt2, pt3):
    # barycentric coordinates on a triangle
    # https://mathworld.wolfram.com/BarycentricCoordinates.html
    # https://www.youtube.com/watch?v=HYAgJN3x4GA
    s, t = sorted([random.random(), random.random()])
    return s * pt1 + (t - s) * pt2 + (1 - t) * pt3


class PointSampler:
    def __init__(self, num_points):
        self.num_points = num_points

    def __call__(self, mesh):
        verts, faces, path = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))
        for i in range(len(areas)):
            areas[i] = triangle_area(verts[faces[i][0]], verts[faces[i][1]], verts[faces[i][2]])
        sampled_faces = random.choices(faces, weights=areas, cum_weights=None, k=self.num_points)
        sampled_points = np.zeros((self.num_points, 3))
        for i in range(len(sampled_faces)):
            sampled_points[i] = sample_point(
                verts[sampled_faces[i][0]], verts[sampled_faces[i][1]], verts[sampled_faces[i][2]]
            )
        np.save(path[:-4], sampled_points.squeeze())
        return sampled_points


class RawLoader(DatasetFolder):
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=read_off,
        is_valid_file=None,
    ):
        super().__init__(
            root,
            loader,
            EXT if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )


def main(
    ext=".ply", num_points=1024, data_root="./data/ModelNet40", sampler_bsize=32, num_workers=4
):
    """
    Create new data in 'BINARY' format from raw data meshes and 'DELETE' mesh files.
    Sample each mesh NUM_POINTS times.
    """
    loader = read_ply if ext == ".ply" else read_off
    dataset_train = RawLoader(
        root=data_root + "train", loader=loader, transform=PointSampler(num_points)
    )
    dataloader_train = DataLoader(
        dataset_train, batch_size=sampler_bsize, num_workers=num_workers
    )
    paths = dataloader_train.dataset.samples

    for batch in tqdm(dataloader_train):
        pass
        # points, targets = batch
        # batch_size = points.shape[0]
        # sample_names = [path[0][:-3] + "npy" for path in paths]
        # [
        #     np.save(sample_name, point.squeeze())
        #     for sample_name, point in zip(sample_names, points)
        # ]
    [os.remove(file) for file in glob(data_root + f"train/*/*{ext}")]

    dataset_test = RawLoader(
        root=data_root + "test", loader=loader, transform=PointSampler(num_points)
    )
    dataloader_test = DataLoader(dataset_test, batch_size=sampler_bsize, num_workers=num_workers)
    paths = dataloader_test.dataset.samples

    for batch in tqdm(dataloader_test):
        pass
        # points, targets = batch
        # batch_size = points.shape[0]
        # sample_names = [path[0][:-3] + "npy" for path in paths]
        # [
        #     np.save(sample_name, point.squeeze())
        #     for sample_name, point in zip(sample_names, points)
        # ]
    [os.remove(file) for file in glob(data_root + f"test/*/*{ext}")]


if __name__ == "__main__":

    with open("config.json", "r") as f:
        cfg = json.load(f)
    main(**cfg["sampler"])
