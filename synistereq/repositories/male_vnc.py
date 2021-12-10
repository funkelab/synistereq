from synistereq.datasets import MaleVnc
from synistereq.interfaces import Neuprint
from .repository import Repository

MaleVncNeuprint = lambda: Repository.for_service_constructor(
    MaleVnc(),
    Neuprint,
    server="neuprint-pre.janelia.org",
    dataset="vnc",
)
