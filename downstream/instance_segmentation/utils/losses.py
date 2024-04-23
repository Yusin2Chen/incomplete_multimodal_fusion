import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np

# invariance loss
sim_loss = nn.MSELoss()

# variance loss
def std_loss(z_a, z_b):
    std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
    std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
    std_loss = torch.mean(F.relu(1 - std_z_a)) + torch.mean(F.relu(1 - std_z_b))
    return std_loss


#function taken from https://github.com/facebookresearch/barlowtwins/tree/a655214c76c97d0150277b85d16e69328ea52fd9
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

# covariance loss
def cov_loss(z_a, z_b):
    N = z_a.shape[0]
    D = z_a.shape[1]
    z_a = z_a - z_a.mean(dim=0)
    z_b = z_b - z_b.mean(dim=0)
    cov_z_a = (z_a.T @ z_a) / (N - 1)
    cov_z_b = (z_b.T @ z_b) / (N - 1)
    cov_loss = off_diagonal(cov_z_a).pow_(2).sum() / D + off_diagonal(cov_z_b).pow_(2).sum() / D
    return cov_loss

def vicreg(repr_a, repr_b, l=25, mu=25, nu=1):

    _sim_loss = sim_loss(repr_a, repr_b)
    _std_loss = std_loss(repr_a, repr_b)
    _cov_loss = cov_loss(repr_a, repr_b)

    loss = l * _sim_loss + mu * _std_loss + nu * _cov_loss

    return loss

class HardNegtive_loss(torch.nn.Module):

    def __init__(self, tau_plus=0.1, beta=1.0, temperature=0.5, alpha=256, estimator='hard'):
        super(HardNegtive_loss, self).__init__()
        self.tau_plus = tau_plus
        self.beta = beta
        self.temperature = temperature
        self.estimator = estimator
        self.alpha = alpha

    def get_negative_mask(self, batch_size):
        negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0
            negative_mask[i, i + batch_size] = 0

        negative_mask = torch.cat((negative_mask, negative_mask), 0)
        return negative_mask

    def forward(self, out_1, out_2):
        # normalization
        batch_size, c = out_1.shape
        out_1 = F.normalize(out_1, dim=1)
        out_2 = F.normalize(out_2, dim=1)
        # neg score
        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        # old_neg = neg.clone()
        mask = self.get_negative_mask(batch_size).cuda()
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        # negative samples similarity scoring
        if self.estimator == 'hard':
            N = batch_size * 2 - 2
            imp = (self.beta * neg.log()).exp()
            reweight_neg = (imp * neg).sum(dim=-1) / imp.mean(dim=-1)
            Ng = (-self.tau_plus * N * pos + reweight_neg) / (1 - self.tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min=N * np.e ** (-1 / self.temperature))
        elif self.estimator == 'easy':
            Ng = neg.sum(dim=-1)
        else:
            raise Exception('Invalid estimator selected. Please use any of [hard, easy]')

        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng))).mean()
        # eqco
        #print(batch_size, Ng.shape)
        #loss = (- torch.log(pos / (pos + self.alpha / Ng.shape[0] * Ng))).mean()

        return loss

class DINOLoss(nn.Module):
    r"""DINOLoss Class
    Compute the loss.
    """

    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9) -> None:
        super(DINOLoss, self).__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        student_output = F.normalize(student_output, dim=1)
        teacher_output = F.normalize(teacher_output, dim=1)
        student_out = [
            F.log_softmax(s / self.student_temp, dim=-1) for s in student_output
        ]
        teacher_out = [
            F.softmax((t - self.center) / self.teacher_temp, dim=-1).detach()
            for t in teacher_output
        ]

        total_loss = 0
        n_loss_terms = 0
        for t_idx, t in enumerate(teacher_out):
            for s_idx, s in enumerate(student_out):
                # Skip for the same image
                if t_idx == s_idx:
                    continue
                loss = torch.sum(-t * s, dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        r"""Update center for teacher output
        Exponential moving average as described in the paper
        """
        batch_center = torch.cat(teacher_output).mean(dim=0, keepdim=True)
        self.center = (
            self.center * self.center_momentum
            + (1 - self.center_momentum) * batch_center
        )

def byol_loss_func(p, z, simplified=True):
    if simplified:
        return 2 - 2 * F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)

        return 2 - 2 * (p * z.detach()).sum(dim=1).mean()
