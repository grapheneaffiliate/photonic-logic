# Utility subpackage for switching/threshold functions and statistics
from .switching import hard_logic, sigmoid, soft_logic, softplus
from .statistics import extract_cascade_statistics, validate_extinction_mode_flags, format_extinction_summary

__all__ = ["sigmoid", "softplus", "soft_logic", "hard_logic", "extract_cascade_statistics", "validate_extinction_mode_flags", "format_extinction_summary"]
