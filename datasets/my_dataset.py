import os, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
import open3d as o3d


class MyTestDataset(Dataset):
    def __init__(self, src_dir, model_dir, n_points=1024):
        self.src_files = sorted(os.listdir(src_dir))
        self.model_files = sorted(os.listdir(model_dir))
        self.src_dir = src_dir
        self.model_dir = model_dir
        self.n_points = n_points

    def load_ply(self, path):
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points)
        if pts.shape[0] > self.n_points:
            idx = np.random.choice(pts.shape[0], self.n_points, replace=False)
            pts = pts[idx]
        return pts.astype(np.float32)

    def __len__(self):
        return len(self.src_files)

    def __getitem__(self, idx):
        src = self.load_ply(os.path.join(self.src_dir, self.src_files[idx]))
        model = self.load_ply(os.path.join(self.model_dir, self.model_files[idx]))

        res = {
            "src_pcd": torch.tensor(src, dtype=torch.float32),
            "model_pcd": torch.tensor(model, dtype=torch.float32),
            "src_pcd_normal": torch.zeros_like(torch.tensor(src, dtype=torch.float32)),
            "model_pcd_normal": torch.zeros_like(torch.tensor(model, dtype=torch.float32)),
        }
        return res


def get_dataset_mytest(option, batch_size=1, core_count=4):
    dataset = MyTestDataset(option.src_dir, option.model_dir, n_points=1024)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=core_count, pin_memory=True)
    return loader, dataset
