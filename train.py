from engine.schedule.scheduler import BaseScheduler
from lanenet import *

if __name__ == '__main__':
    BaseScheduler().schedule()


    import torch.optim.lr_scheduler
    torch.optim.lr_scheduler.CosineAnnealingLR()