# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.1.35"

# 简化导入，避免循环依赖
try:
    from ultralytics.data.explorer.explorer import Explorer
except ImportError:
    Explorer = None

from ultralytics.models import RTDETR, SAM, YOLO, YOLOWorld
from ultralytics.models.fastsam import FastSAM
from ultralytics.models.nas import NAS
from ultralytics.utils import ASSETS, SETTINGS as settings
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
)

if Explorer is not None:
    __all__ += ("Explorer",)
