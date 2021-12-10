from .fafb_model import FafbModel
from .hemi_model import HemiModel
from .male_vnc_model import MaleVncModel

KNOWN_MODELS = {
    "FAFB": FafbModel,
    "HEMI": HemiModel,
    "MALE_VNC": MaleVncModel,
}
