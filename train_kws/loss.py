import math

import torch
from torch import nn


class ArcMarginLoss(nn.Module):
    """
    Implement of additive angular margin loss.
    """

    def __init__(self,
                 scale=32.0,
                 margin=0.2,
                 easy_margin=False):
        super(ArcMarginLoss, self).__init__()
        self.scale = scale
        self.easy_margin = easy_margin
        self.criterion = nn.CrossEntropyLoss()

        self.update(margin)

    def forward(self, cosine, label):
        # cosine : [batch, numclasses].
        # label : [batch, ].
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mmm)

        one_hot = torch.zeros(cosine.size()).type_as(cosine)
        one_hot.scatter_(1, label.unsqueeze(1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        loss = self.criterion(output, label)
        return loss

    def update(self, margin=0.2):
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.m = self.margin
        self.mmm = 1.0 + math.cos(math.pi - margin)


class MarginScheduler:
    def __init__(
            self,
            criterion,
            increase_start_epoch,
            fix_epoch,
            step_per_epoch,
            initial_margin,
            final_margin,
            increase_type='exp',
    ):
        assert hasattr(criterion, 'update'), "Loss function not has 'update()' attributes."
        self.criterion = criterion
        self.increase_start_step = increase_start_epoch * step_per_epoch
        self.fix_step = fix_epoch * step_per_epoch
        self.initial_margin = initial_margin
        self.final_margin = final_margin
        self.increase_type = increase_type
        self.margin = initial_margin

        self.current_step = 0
        self.increase_step = self.fix_step - self.increase_start_step

        self.init_margin()

    def init_margin(self):
        self.criterion.update(margin=self.initial_margin)

    def step(self, current_step=None):
        if current_step is not None:
            self.current_step = current_step

        self.margin = self.iter_margin()
        self.criterion.update(margin=self.margin)
        self.current_step += 1

    def iter_margin(self):
        if self.current_step < self.increase_start_step:
            return self.initial_margin

        if self.current_step >= self.fix_step:
            return self.final_margin

        a = 1.0
        b = 1e-3

        current_step = self.current_step - self.increase_start_step
        if self.increase_type == 'exp':
            # exponentially increase the margin
            ratio = 1.0 - math.exp(
                (current_step / self.increase_step) *
                math.log(b / (a + 1e-6))) * a
        else:
            # linearly increase the margin
            ratio = 1.0 * current_step / self.increase_step
        return self.initial_margin + (self.final_margin -
                                      self.initial_margin) * ratio

    def get_margin(self):
        return self.margin
