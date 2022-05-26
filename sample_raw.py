import os
import torch
import random
import numpy as np
from tqdm import tqdm
from plyfile import PlyData
from torch.utils.data import DataLoader
from torchvision.datasets.folder import DatasetFolder

EXT = (".ply", ".off")


def read_ply(path):

    with open(path, "rb") as f:
        data = PlyData.read(f)
    verts = [torch.tensor(data["vertex"][axis]) for axis in ["x", "y", "z"]]
    verts = torch.stack(verts, dim=-1)
    faces = None
    if "face" in data:
        faces = data["face"]["vertex_indices"]
        faces = [torch.tensor(face, dtype=torch.long) for face in faces]
        faces = torch.stack(faces)

    return verts, faces


def read_ply_native(path):

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

    verts = [[float(s) for s in file.readline().strip().split(" ")] for i_vert in range(n_verts)]
    faces = [
        [int(s) for s in file.readline().strip().split(" ")][1:] for i_face in range(n_faces)
    ]
    file.close()
    return torch.tensor(verts), torch.tensor(faces)


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


class PointSampler:
    def __init__(self, num_points):
        self.num_points = num_points

    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        # Heron's Formula for the triangle area.
        u = 0.5 * (side_a + side_b + side_c)
        return max(u * (u - side_a) * (u - side_b) * (u - side_c), 0) ** 0.5

    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        # https://www.youtube.com/watch?v=HYAgJN3x4GA
        s, t = sorted([random.random(), random.random()])
        return s * pt1 + (t - s) * pt2 + (1 - t) * pt3

    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))
        for i in range(len(areas)):
            areas[i] = self.triangle_area(
                verts[faces[i][0]], verts[faces[i][1]], verts[faces[i][2]]
            )
        sampled_faces = random.choices(faces, weights=areas, cum_weights=None, k=self.num_points)
        sampled_points = np.zeros((self.num_points, 3))
        for i in range(len(sampled_faces)):
            sampled_points[i] = self.sample_point(
                verts[sampled_faces[i][0]], verts[sampled_faces[i][1]], verts[sampled_faces[i][2]]
            )
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


def main():
    """

    Create new data in 'BINARY' format from raw data meshes and 'DELETE' mesh files.
    Sample each mesh NUM_POINTS times.

    """

    NUM_POINTS = 1024
    DATA_ROOT = "./CustomData/"

    dataset_train = RawLoader(
        root=DATA_ROOT + "train", loader=read_off, transform=PointSampler(NUM_POINTS)
    )
    dataloader_train = DataLoader(dataset_train, batch_size=32, num_workers=32)

    for idx, file in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
        points, _ = file
        batch_size = points.shape[0]
        paths = dataloader_train.dataset.samples[idx * batch_size : (idx + 1) * batch_size]
        sample_names = [path[0][:-3] + "npy" for path in paths]
        [
            np.save(sample_name, point.squeeze())
            for sample_name, point in zip(sample_names, points)
        ]

    
    dataset_test = RawLoader(
        root=DATA_ROOT + "test", loader=read_off, transform=PointSampler(NUM_POINTS)
    )
    dataloader_test = DataLoader(dataset_test, batch_size=32, num_workers=32)

    for idx, file in tqdm(enumerate(dataloader_test), total=len(dataloader_test)):
        points, _ = file
        batch_size = points.shape[0]
        paths = dataloader_test.dataset.samples[idx * batch_size : (idx + 1) * batch_size]
        sample_names = [path[0][:-3] + "npy" for path in paths]
        [
            np.save(sample_name, point.squeeze())
            for sample_name, point in zip(sample_names, points)
        ]


if __name__ == "__main__":
    main()
