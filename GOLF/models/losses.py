import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch_geometric.utils import scatter


def l2loss_atomwise(pred, target, reduction="mean"):
    dist = torch.linalg.vector_norm((pred - target), dim=-1)
    if reduction == "mean":
        return torch.mean(dist)
    elif reduction == "sum":
        return torch.sum(dist)
    else:
        return dist


class L2AtomwiseLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(L2AtomwiseLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        loss = l2loss_atomwise(pred, target, self.reduction)
        return loss


class L2Loss(nn.Module):
    def __init__(self) -> None:
        super(L2Loss, self).__init__()

    def forward(self, pred, target, batch):
        return torch.mean(
            scatter(
                mse_loss(pred, target, reduction="none").mean(-1),
                batch.batch,
                dim_size=batch.batch_size,
                reduce="mean",
            )
        )
