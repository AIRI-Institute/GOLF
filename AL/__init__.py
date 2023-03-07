import torch

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
