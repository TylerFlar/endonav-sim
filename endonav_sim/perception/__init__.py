"""Perception subpackage: extract navigation signals from rendered RGB frames."""

from .junction import Blob, JunctionDetector, JunctionResult
from .place_recognition import MatchResult, PlaceRecognition
from .proximity import ProximityDetector, ProximityResult

__all__ = [
    "Blob",
    "JunctionDetector",
    "JunctionResult",
    "MatchResult",
    "PlaceRecognition",
    "ProximityDetector",
    "ProximityResult",
]
