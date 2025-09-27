import os, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
import open3d as o3d


def farthest_point_sampling(pts, n_samples):
    """
    最远点采样 (Farthest Point Sampling, FPS)
    pts: (N, 3) numpy array
    n_samples: 采样点数
    return: (n_samples, 3) numpy array
    """
    N = pts.shape[0]
    if N <= n_samples:
        return pts

    sampled_idx = np.zeros(n_samples, dtype=np.int64)
    distances = np.ones(N) * 1e10

    # 随机选择第一个点
    farthest = np.random.randint(0, N)
    for i in range(n_samples):
        sampled_idx[i] = farthest
        centroid = pts[farthest, :]
        dist = np.sum((pts - centroid) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        farthest = np.argmax(distances)

    return pts[sampled_idx]


def preprocess_pcd(pts, scale=256.0):
    """
    点云预处理: 中心化 + 缩放
    pts: (N, 3) numpy array
    scale: 缩放因子 (与训练数据保持一致)
    """
    # 中心化
    pts = pts - np.mean(pts, axis=0, keepdims=True)
    # 缩放
    pts = pts / scale
    return pts.astype(np.float32)


class MyTestDataset(Dataset):
    def __init__(self, src_dir, model_dir, n_points=1024, scale=256.0):
        self.src_files = sorted(os.listdir(src_dir))
        self.model_files = sorted(os.listdir(model_dir))
        self.src_dir = src_dir
        self.model_dir = model_dir
        self.n_points = n_points
        self.scale = scale

    def load_ply(self, path):
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points).astype(np.float32)

        # 预处理 (中心化 + 缩放)
        pts = preprocess_pcd(pts, scale=self.scale)

        # 稀疏采样点（给模型用）
        if pts.shape[0] > self.n_points:
            pts_sampled = farthest_point_sampling(pts, self.n_points)
        else:
            pts_sampled = pts

        return pts_sampled, pts

    def __len__(self):
        return len(self.src_files)

    def __getitem__(self, idx):
        # 扫描点云 (src)
        src_sampled, src_full = self.load_ply(os.path.join(self.src_dir, self.src_files[idx]))
        # CAD 模型点云 (model)
        model_sampled, model_full = self.load_ply(os.path.join(self.model_dir, self.model_files[idx]))

        res = {
            # 模型输入用（稀疏点云）
            "src_pcd": torch.tensor(src_sampled, dtype=torch.float32),
            "model_pcd": torch.tensor(model_sampled, dtype=torch.float32),
            "src_pcd_normal": torch.zeros_like(torch.tensor(src_sampled, dtype=torch.float32)),
            "model_pcd_normal": torch.zeros_like(torch.tensor(model_sampled, dtype=torch.float32)),

            # 完整点云（可视化/保存用）
            "src_pcd_full": torch.tensor(src_full, dtype=torch.float32),
            "model_pcd_full": torch.tensor(model_full, dtype=torch.float32),
        }
        return res


def get_dataset_mytest(opts, batch_size=1, n_cores=4):
    dataset = MyTestDataset(opts.src_dir, opts.model_dir, n_points=1024, scale=256.0)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=n_cores, pin_memory=True)
    return loader, dataset
