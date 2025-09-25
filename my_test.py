import torch
import numpy as np, random, torch, pdb
from collections import OrderedDict, defaultdict
from tqdm import tqdm

from utils.options import option
from utils.commons import save_data
from utils.se_math import se3
from utils.diffusion_scheduler import diffusion_scheduler

# ====== 自己的数据集 ======
from datasets.my_dataset import get_dataset_mytest   # 你要把MyTestDataset放在datasets/my_dataset.py

option.seed = 1234
np.random.seed(option.seed)
random.seed(option.seed)
torch.manual_seed(option.seed)
torch.cuda.manual_seed(option.seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def init_option(option):
    option.is_debug_mode = False
    option.test_mode = True
    option.schedule_kind = ["linear", "cosine"][1]

    option.save_results = True
    option.diffusion_stepss = 5
    option.beta_start = 0.2
    option.beta_end = 0.8
    option.noise_level_r = 0.1
    option.noise_level_t = 0.01
    option.enable_noise = True

    return option

# ========== 模型加载 ==========
def get_model(option):
    option.model_type = "DCP"
    option.model_path = "./results/REG-DCP-tudl-diffusion_200_0.00010_0.05_0.05_0.03-nvids3_cosine/model_epoch19.pth"

    from modules.DCP.dcp import DCP
    surrogate_model = DCP(option)
    option.vs = diffusion_scheduler(option)

    try:
        surrogate_model.load_state_dict(
            OrderedDict({k[7:]: v for k, v in torch.load(option.model_path, map_location=option.device).items()}))
    except:
        surrogate_model.load_state_dict(
            OrderedDict({k: v for k, v in torch.load(option.model_path, map_location=option.device).items()}))

    surrogate_model = surrogate_model.to(option.device)
    surrogate_model.eval()
    print("✅ Loaded model:", option.model_path)
    return surrogate_model


# ========== 主测试流程 ==========
def main(option):
    option = init_option(option)
    surrogate_model = get_model(option)

    # 使用自定义数据集
    test_loader, test_db = get_dataset_mytest(option, batch_size=1)

    rcd = defaultdict(list)

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            data = {k: v.to(option.device).float() for k, v in data.items()}
            X, X_normal = data["src_pcd"], data["src_pcd_normal"]
            Y, Y_normal = data["model_pcd"], data["model_pcd_normal"]
            B = len(X)

            H_t = torch.eye(4)[None].expand(B, -1, -1).to(option.device)

            for t in range(option.diffusion_stepss, 1, -1):
                X_t = (H_t[:, :3, :3] @ X.transpose(2, 1) + H_t[:, :3, [3]]).transpose(2, 1)
                X_normal_t = (H_t[:, :3, :3] @ X_normal.transpose(2, 1)).transpose(2, 1)

                Rs_pred, ts_pred = surrogate_model.forward({
                    "src_pcd": X_t,
                    "src_pcd_normal": X_normal_t,
                    "model_pcd": Y,
                    "model_pcd_normal": Y_normal
                })

                _delta_H_t = torch.cat([Rs_pred, ts_pred.unsqueeze(-1)], dim=2)
                delta_H_t = torch.eye(4)[None].expand(B, -1, -1).to(option.device)
                delta_H_t[:, :3, :] = _delta_H_t
                H_0 = delta_H_t @ H_t

                gamma0 = option.vs.gamma0[t]
                gamma1 = option.vs.gamma1[t]
                H_t = se3.exp(gamma0 * se3.log(H_0) + gamma1 * se3.log(H_t))

            Rs_pred = H_0[:, :3, :3]
            ts_pred = H_0[:, :3, 3]

            Rs_pred = torch.inverse(Rs_pred)
            ts_pred = (- Rs_pred @ ts_pred[:, :, None])[:, :, 0]

            # 保存预测结果到字典
            rcd["Rs_pred"].extend(list(Rs_pred.cpu().numpy()))
            rcd["ts_pred"].extend(list(ts_pred.cpu().numpy()))
            rcd["src_pcd"].extend(list(X.cpu().numpy()))
            rcd["model_pcd"].extend(list(Y.cpu().numpy()))
            rcd["src_pcd_normal"].extend(list(X_normal.cpu().numpy()))
            rcd["model_pcd_normal"].extend(list(Y_normal.cpu().numpy()))

    # ===== 保存结果字典 =====
    if option.save_results:
        save_path = f"./results/mytest_results2_T{option.diffusion_stepss}.pth"
        save_data(save_path, rcd)
        print(f"✅ 测试完成，预测结果已保存到: {save_path}")


if __name__ == '__main__':
    # 基本配置
    option.device = "cuda" if torch.cuda.is_available() else "cpu"
    option.diffusion_stepss = 5
    option.save_results = True

    # 数据路径
    option.src_dir = "./my_dataset/group2/src"
    option.model_dir = "./my_dataset/group2/model"

    main(option)
