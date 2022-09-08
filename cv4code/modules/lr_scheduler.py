# SPDX-License-Identifier: Apache-2.0
from torch.optim import lr_scheduler
from warmup_scheduler import GradualWarmupScheduler

class StepLRScheduler:
    def __init__(
        self,
        optim,
        base_lr,
        step_size,
        gamma,
        warmup_steps=0,
        steps_per_epoch=None,
        **kargs
    ):
        super(StepLRScheduler, self).__init__()
        if steps_per_epoch and steps_per_epoch > 0:
            step_size = step_size * steps_per_epoch
            warmup_steps = warmup_steps * steps_per_epoch
        self.optim = optim
        self.base_lr = base_lr
        self.step_size = step_size
        self.gamma = gamma
        self.scheduler = lr_scheduler.StepLR(optim, step_size=step_size, gamma=gamma)
        if warmup_steps > 0:
            self.scheduler = GradualWarmupScheduler(
                optim, 1.0, warmup_steps, self.scheduler
            )
            self.optim.step()
    
    def step(self):
        self.scheduler.step()
    
    def get_last_lr(self):
        return self.optim.param_groups[0]["lr"]

class CosineAnnealingLRScheduler:
    def __init__(
        self,
        optim,
        base_lr,
        steps_per_epoch,
        total_epochs,
        warmup_steps=0,
        **kargs
    ):
        super(CosineAnnealingLRScheduler, self).__init__()
        if steps_per_epoch and steps_per_epoch > 0:
            warmup_steps = warmup_steps * steps_per_epoch
        self.optim = optim
        self.base_lr = base_lr
        t_max = total_epochs * steps_per_epoch
        self.scheduler = lr_scheduler.CosineAnnealingLR(optim, T_max=t_max)
        if warmup_steps > 0:
            self.scheduler = GradualWarmupScheduler(
                optim, 1.0, warmup_steps, self.scheduler
            )
            self.optim.step()
    
    def step(self):
        self.scheduler.step()
    
    def get_last_lr(self):
        return self.optim.param_groups[0]["lr"]