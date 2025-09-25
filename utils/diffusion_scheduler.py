import torch
import numpy as np

class diffusion_scheduler(torch.nn.Module):
    def __init__(self, option):
        super().__init__()
        
        # ✅ 和 test.py 保持一致
        self.num_steps = option.diffusion_steps
        self.beta_start = option.beta_start
        self.beta_end = option.beta_end
        self.mode = option.scheduler_variant
        self.S = option.S   # 直接使用 test.py 里传进来的 S

        if self.mode == 'linear':
            # 线性调度
            betas = torch.linspace(self.beta_start, self.beta_end, steps=self.num_steps)

        elif self.mode == 'cosine':
            # 余弦调度
            def betas_fn(s):
                T = self.num_steps
                def f(t, T):
                    return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2

                alphas = []
                f0 = f(0, T)
                for t in range(T + 1):
                    alphas.append(f(t, T) / f0)

                betas = []
                for t in range(1, T + 1):
                    betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))
                return betas

            betas = torch.FloatTensor(betas_fn(self.S))

        else:
            raise NotImplementedError(f"Unknown scheduler mode: {self.mode}")

        # ✅ padding，保持形状一致
        self.betas = torch.cat([torch.zeros([1]), betas], dim=0)
        self.alphas = 1 - self.betas

        log_alphas = torch.log(self.alphas)
        for i in range(1, log_alphas.size(0)):
            log_alphas[i] += log_alphas[i - 1]
        self.alpha_bars = log_alphas.exp()

        # γ 系数 (用于公式里的插值)
        self.gamma0 = torch.zeros_like(self.betas)
        self.gamma1 = torch.zeros_like(self.betas)
        self.gamma2 = torch.zeros_like(self.betas)
        for t in range(2, self.num_steps + 1):
            self.gamma0[t] = self.betas[t] * torch.sqrt(self.alpha_bars[t - 1]) / (1. - self.alpha_bars[t])
            self.gamma1[t] = (1. - self.alpha_bars[t - 1]) * torch.sqrt(self.alphas[t]) / (1. - self.alpha_bars[t])
            self.gamma2[t] = (1. - self.alpha_bars[t - 1]) * self.betas[t] / (1. - self.alpha_bars[t])

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps + 1), batch_size)
        return ts.tolist()
