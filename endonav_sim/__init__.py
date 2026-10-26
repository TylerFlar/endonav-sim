"""endonav_sim: procedural endoscopic kidney simulator with SOFA physics."""

from .anatomy import AnatomyMeta, AnatomyParams, generate_anatomy
from .dynamics import CommandFeedback, ScopeLimits
from .simulator import CaptureResult, CommandResult, KidneySimulator, ToolMode
from .stones import Stone, StoneParams, generate_stones
from .visuals import ToolVisualState, compose_visuals

__version__ = "0.8.1"
__all__ = [
    "KidneySimulator",
    "AnatomyParams",
    "AnatomyMeta",
    "generate_anatomy",
    "StoneParams",
    "Stone",
    "generate_stones",
    "ScopeLimits",
    "CommandFeedback",
    "CommandResult",
    "ToolMode",
    "CaptureResult",
    "ToolVisualState",
    "compose_visuals",
]
