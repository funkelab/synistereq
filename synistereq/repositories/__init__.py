from synistereq.datasets import *
from synistereq.interfaces import *

from .fafb import FafbFlywire, FafbCatmaid
from .hemi import HemiNeuprint
from .male_vnc import MaleVncNeuprint
from .repository import Repository


KNOWN_REPOSITORIES = {
    "FAFB_CATMAID": FafbCatmaid,
    "FAFB_FLYWIRE": FafbFlywire,
    "HEMI_NEUPRINT": HemiNeuprint,
    "MALE_VNC_NEUPRINT": MaleVncNeuprint,
}
