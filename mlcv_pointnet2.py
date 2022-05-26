#!/usr/bin/env python
# coding: utf-8

# ## Dataset and Dataloader

# In[1]:

import torch
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets.folder import DatasetFolder

USE_CPU = False
NUM_EPOCHS = 5
BATCH_SIZE = 128


# In[2]:


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


class PointSampler(object):
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


# EXT = (".off", ".ply",)
EXT = (".npy", )

class CustomDS(DatasetFolder):
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=None,
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


# In[3]:

class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2
        
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0) 
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return  norm_pointcloud

class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        return torch.from_numpy(pointcloud)

data_path = "./ModelNet40_Binary/"

transform = transforms.Compose([
    Normalize(),
    ToTensor(),
    ])

train_dataset = CustomDS(root=data_path + "train", loader=np.load, transform=transform)
test_dataset = CustomDS(root=data_path + "test", loader=np.load, transform=transform)
trainDataLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)
testDataLoader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=10)


# In[4]:


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat(
            [grouped_xyz_norm, grouped_points], dim=-1
        )  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    """batch_pc: BxNx3"""
    for b in range(batch_pc.shape[0]):
        dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            batch_pc[b, drop_idx, :] = batch_pc[b, 0, :]  # set to the first point
    return batch_pc

def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """Randomly scale the point cloud. Scale is per point cloud.
    Input:
        BxNx3 array, original batch of point clouds
    Return:
        BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data

def shift_point_cloud(batch_data, shift_range=0.1):
    """Randomly shift point cloud. Shift is per point cloud.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
    for batch_index in range(B):
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    return batch_data

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    )
    assert (0 <= batch_indices.min()) and (batch_indices.max() <= 1023)
    assert (0 <= idx.min()) and (idx.max() <= 1023)
    new_points = points[batch_indices, idx, :]
    return new_points

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


# In[5]:


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points
            )
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


# In[6]:


class PointNet(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super().__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        return x, l3_points


# In[7]:


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss


# In[8]:


classifier = PointNet(num_class=40, normal_channel=False).to('cuda')


# In[9]:


optimizer = torch.optim.Adam(
    classifier.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
criterion = Loss()


# In[10]:


for epoch in range(NUM_EPOCHS):
    mean_correct = []
    classifier = classifier.train()
    scheduler.step()

    for batch_id, (points, target) in tqdm(
        enumerate(trainDataLoader), total=len(trainDataLoader)
    ):
        optimizer.zero_grad()
        points = points.squeeze(1)
        points = points.data.numpy()
        points = random_point_dropout(points)
        points[:, :, 0:3] = random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = shift_point_cloud(points[:, :, 0:3])
        points = torch.Tensor(points)
        points = points.transpose(2, 1)

        if not USE_CPU:
            points, target = points.cuda(), target.cuda()

        pred, trans_feat = classifier(points)
        loss = criterion(pred, target.long(), trans_feat)
        pred_choice = pred.data.max(1)[1]

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))
        loss.backward()
        optimizer.step()

    train_instance_acc = np.mean(mean_correct)
    print(f"Train instance accuracy: {train_instance_acc:.4f}")


# In[ ]:


points[1]


# In[ ]:




