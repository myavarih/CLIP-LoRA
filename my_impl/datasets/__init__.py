from .oxford_pets import OxfordPets
from .eurosat import EuroSAT
from .ucf101 import UCF101
from .sun397 import SUN397
from .caltech101 import Caltech101
from .dtd import DescribableTextures
from .fgvc import FGVCAircraft
from .food101 import Food101
from .oxford_flowers import OxfordFlowers
from .stanford_cars import StanfordCars
from .imagenet import ImageNet
from .walnut import Walnut
from .badam_zamini import BadamZamini
from .date_rabi import DateRabi
from .fig_gonjeshk import FigGonjeshk
from .lobia_ghermez import LobiaGhermez
from .nokhod import Nokhod
from .piarom_color import PiaromColor
from .piarom_shape import PiaromShape
from .pistachio_afat import PistachioAfat
from .pistachio_poost_maghz import PistachioPoostMaghz


dataset_list = {
    # Benchmark datasets
    "oxford_pets": OxfordPets,
    "eurosat": EuroSAT,
    "ucf101": UCF101,
    "sun397": SUN397,
    "caltech101": Caltech101,
    "dtd": DescribableTextures,
    "fgvc": FGVCAircraft,
    "food101": Food101,
    "oxford_flowers": OxfordFlowers,
    "stanford_cars": StanfordCars,
    "imagenet": ImageNet,

    # Few-shot fine-grained agricultural datasets
    "walnut": Walnut,
    "Walnut_Color_Parvizi_3": Walnut,

    "badam_zamini": BadamZamini,
    "peanut": BadamZamini,
    "BadamZamini_Chitgar_2": BadamZamini,

    "date_rabi": DateRabi,
    "Date_Rabi_Tarom_2": DateRabi,

    "fig_gonjeshk": FigGonjeshk,
    "fig": FigGonjeshk,
    "Fig_Gonjeshk_12": FigGonjeshk,

    "lobia_ghermez": LobiaGhermez,
    "red_bean": LobiaGhermez,
    "LobiaGhermez_gharebaghi_1": LobiaGhermez,

    "nokhod": Nokhod,
    "chickpea": Nokhod,
    "Nokhod_Gherabaghi_1": Nokhod,

    "piarom_color": PiaromColor,
    "Piarom_Color_Ameri_1": PiaromColor,

    "piarom_shape": PiaromShape,
    "Piarom_Shape_Ameri_3": PiaromShape,

    "pistachio_afat": PistachioAfat,
    "pistachio_pest": PistachioAfat,
    "Pistachio_AfatAbdollahzadeh_1": PistachioAfat,

    "pistachio_poost_maghz": PistachioPoostMaghz,
    "pistachio_hull": PistachioPoostMaghz,
    "Pistachio_poost maghz_12": PistachioPoostMaghz,
}


def build_dataset(dataset, root_path, shots, preprocess=None):
    if dataset == 'imagenet':
        return dataset_list[dataset](root_path, shots, preprocess)
    else:
        return dataset_list[dataset](root_path, shots)