from utils.utils import ignore_extra_args

from .dimenetplusplus import DimeNetPlusPlusPotential
from .dimenetplusplus_oc import DimeNetPlusPlusWrap
from .gemnet_oc import GemNetOC
from .painn import PaiNN

NeuralNetworkPotentials = {
    "DimenetPlusPlus": ignore_extra_args(DimeNetPlusPlusPotential),
    "DimenetPlusPlusOC": ignore_extra_args(DimeNetPlusPlusWrap),
    "PaiNN": ignore_extra_args(PaiNN),
    "GemNetOC": ignore_extra_args(GemNetOC),
}
