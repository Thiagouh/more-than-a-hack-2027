"""Frente 5: ball-event evidence; never a final possession state machine."""
from .contract import DEFAULTS, VERSION, EVENT_VERSION
from .detector import Detector, detect

__all__ = ['DEFAULTS', 'VERSION', 'Detector', 'detect', 'EVENT_VERSION']
