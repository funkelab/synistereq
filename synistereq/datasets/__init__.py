from .fafb import Fafb
from .hemi import Hemi
from .male_vnc import MaleVnc

KNOWN_DATASETS = {
    "FAFB": Fafb,
    "HEMI": Hemi,
    "MALE_VNC": MaleVnc,
}
