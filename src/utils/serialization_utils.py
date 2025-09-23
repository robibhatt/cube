import torch
from typing import Tuple, Any, Optional
from pathlib import Path

def encode_shape(shape: Optional[torch.Size]) -> Optional[Tuple[int, ...]]:
    if shape is None:
        return None
    return tuple(shape)

def decode_shape(data: Any) -> Optional[torch.Size]:
    if data is None:
        return None
    return torch.Size(data)

def encode_dtype(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")

def decode_dtype(name: str) -> torch.dtype:
    return getattr(torch, name)

def encode_path(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    return str(path)

def decode_path(data: Optional[str]) -> Optional[Path]:
    if data is None:
        return None
    return Path(data)