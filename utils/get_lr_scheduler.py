import torch.optim.lr_scheduler as lr_scheduler

def get_lr_scheduler(option, optimizer):
    scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.2)
    return scheduler
