"""Fourier analysis utilities for MLP-style models."""

from .fourier_mlp_module import FourierMlp
from .mlp_fourier_detector import detect_mlp_fourier_components

__all__ = ["FourierMlp", "detect_mlp_fourier_components"]
