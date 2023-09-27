import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KCALMOL2HARTREE = 627.5
PSI4_BOHR2ANGSTROM = 0.52917720859
