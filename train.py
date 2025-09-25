import os, numpy as np, random, torch, argparse, pdb
from utils.options import option
from utils.diffusion_scheduler import diffusion_scheduler
from utils.get_lr_scheduler import get_lr_scheduler
from tqdm import tqdm
from collections import defaultdict
from utils.se_math import se3
from datasets.get_dataset import GetDataset
from utils.losses import compute_losses, compute_losses_diff

np.random.seed(option.seed)
random.seed(option.seed)
torch.manual_seed(option.seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = False


def InitializeOptions(option):
    option.is_debug_mode = False
    option.model_name = "REG"
    option.testing_mode = False
    option.normalization_enabled = True
    option.core_count = 6
    option.scheduler_variant = ["linear", "cosine"][1]
    option.S = 0.008

    # dataset config
    if option.db_nm == "tudl":
        option.video_count = 3
        option.n_epoches, option.n_start_epoches, option.batch_size = 20, 0, 32
        option.video_infos = ["000001", "000002", "000003"]
    else:
        raise NotImplementedError

    # diffusion configuration
    option.diffusion_steps = 200
    option.beta_start = 1e-4
    option.beta_end = 0.05
    option.noise_level_r = 0.05
    option.noise_level_t = 0.03
    diffusion_str = f"diffusion_{option.diffusion_steps}_{option.beta_start:.5f}_{option.beta_end:.2f}_{option.noise_level_r:.2f}_{option.noise_level_t:.2f}"

    option.results_dir = f"./results/{option.model_name}-{option.net_type}-{option.db_nm}-{diffusion_str}-nvids{option.video_count}_{option.scheduler_variant}"
    print(option.results_dir)
    return option


def main(option):

    option = InitializeOptions(option)

    # model setting
    from modules.DCP.dcp import DCP
    option.vs = diffusion_scheduler(option)
    surrogate_model = DCP(option)

    if torch.cuda.device_count() > 1:
        surrogate_model = torch.nn.DataParallel(surrogate_model, range(torch.cuda.device_count()))
    surrogate_model = surrogate_model.to(option.device)

    train_loader, train_db = GetDataset(
        option,
        db_nm=option.db_nm,
        cls_nm=option.video_infos,
        partition="train",
        batch_size=option.batch_size,
        shuffle=True,
        drop_last=True,
        core_count=option.core_count
    )

    optimizer = torch.optim.Adam(surrogate_model.parameters(), lr=option.lr, betas=(0.9, 0.999))
    scheduler = get_lr_scheduler(option, optimizer)
    cal = lambda x: np.mean(x).item()

    # training
    for epoch_index in range(option.n_epoches):

        # train
        surrogate_model.train()

        rcd = defaultdict(list)
        for i, data in enumerate(tqdm(train_loader, 0)):

            data = {k: v.to(option.device) for k, v in data.items()}

            # model prediction
            X, X_normal = data["src_pcd"], data["src_pcd_normal"]  # [B, N, 3]
            Y, Y_normal = data["model_pcd"], data["model_pcd_normal"]  # [B, M, 3]
            Rs_gt, ts_gt = data['transform_gt'][:, :3, :3], data["transform_gt"][:, :3, 3]  # [B, 3, 3], [B, 3]
            B = Rs_gt.shape[0]

            # SE(3) diffusion process
            H_0 = torch.eye(4)[None].expand(B, -1, -1).to(option.device)
            H_0[:, :3, :3], H_0[:, :3, 3] = Rs_gt, ts_gt
            H_T = torch.eye(4)[None].expand(B, -1, -1).to(option.device)

            taus = option.vs.uniform_sample_t(B)
            alpha_bars = option.vs.alpha_bars[taus].to(option.device)[:, None]  # [B, 1]
            H_t = se3.exp((1. - torch.sqrt(alpha_bars)) * se3.log(H_T @ torch.inverse(H_0))) @ H_0

            # add noise
            scale = torch.cat([torch.ones(3) * option.noise_level_r, torch.ones(3) * option.noise_level_t])[None].to(option.device)  # [1, 6]
            noise = torch.sqrt(1. - alpha_bars) * scale * torch.randn(B, 6).to(option.device)  # [B, 6]
            H_noise = se3.exp(noise)
            H_t_noise = H_noise @ H_t  # [B, 4, 4]

            T_t_R = H_t_noise[:, :3, :3]  # [B, 3, 3]
            T_t_t = H_t_noise[:, :3, 3]  # [B, 3]

            X_t = (T_t_R @ X.transpose(2, 1) + T_t_t.unsqueeze(-1)).transpose(2, 1)  # [B, N, 3]
            X_normal_t = (T_t_R @ X_normal.transpose(2, 1)).transpose(2, 1)          # [B, N, 3]

            transform_gt = torch.eye(4)[None].expand(B, -1, -1).to(option.device)
            transform_gt[:, :3] = data['transform_gt']
            input = {
                "src_pcd": X_t,
                "src_pcd_normal": X_normal_t,
                "model_pcd": Y,
                "model_pcd_normal": Y_normal,
            }
            Rs_pred_rot, ts_pred_rot = surrogate_model.forward(input)
            pred_transforms = torch.cat([Rs_pred_rot, ts_pred_rot.unsqueeze(-1)], dim=2)  # [B, 3, 4]
            train_losses_diff = compute_losses_diff(option, X, X_t, [pred_transforms], data['transform_gt'], loss_type="mae", reduction='mean')
            loss = train_losses_diff['total']

            # original loss
            input = {
                "src_pcd": X,
                "src_pcd_normal": X_normal,
                "model_pcd": Y,
                "model_pcd_normal": Y_normal,
            }
            Rs_pred_rot1, ts_pred_rot1 = surrogate_model.forward(input)
            pred_transforms1 = torch.cat([Rs_pred_rot1, ts_pred_rot1.unsqueeze(-1)], dim=2)  # [B, 3, 4]
            train_losses_origin = compute_losses(option, X, [pred_transforms1], data['transform_gt'], loss_type="mae", reduction='mean')
            loss += train_losses_origin["total"]
            rcd["losses"].append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("=== Train. Epoch [%d], losses: %1.3f ===" % (epoch_index, cal(rcd["losses"])))

            if i > 0 and i % 200 == 0 and not option.is_debug_mode:
                print("Save model. %s" % ('%s/model_epoch%d.pth' % (option.results_dir, epoch_index)))
                torch.save(surrogate_model.state_dict(), '%s/model_epoch%d.pth' % (option.results_dir, epoch_index))

        print(option.results_dir)

        # save model
        if not option.is_debug_mode:
            print("Save model. %s" % ('%s/model_epoch%d.pth' % (option.results_dir, epoch_index)))
            torch.save(surrogate_model.state_dict(), '%s/model_epoch%d.pth' % (option.results_dir, epoch_index))
        else:
            print("Debug. Not save model.")

        scheduler.step()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--net_type', default="DCP", type=str, choices=['DCP'])
    parser.add_argument('--db_nm', default="tudl", type=str, choices=["tudl"])
    args = parser.parse_args()

    option.net_type = args.net_type
    option.db_nm = args.db_nm
    main(option)
