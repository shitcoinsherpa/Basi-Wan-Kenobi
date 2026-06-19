# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from . import configs, distributed, modules

try:
    from .image2video import WanI2V
except (ModuleNotFoundError, ImportError):
    WanI2V = None

try:
    from .speech2video import WanS2V
except (ModuleNotFoundError, ImportError):
    WanS2V = None

try:
    from .text2video import WanT2V
except (ModuleNotFoundError, ImportError):
    WanT2V = None

try:
    from .textimage2video import WanTI2V
except (ModuleNotFoundError, ImportError):
    WanTI2V = None

try:
    from .animate import WanAnimate
except (ModuleNotFoundError, ImportError):
    WanAnimate = None
