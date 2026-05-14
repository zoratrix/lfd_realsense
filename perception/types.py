from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class DetectedObject:
    class_name: str          # 'cup_silver', 'cube_blue', ...
    confidence: float
    bbox: tuple              # (x, y, w, h) в пикселях
    mask: Optional[np.ndarray] = None   # H×W bool, если YOLOv5-seg
    center_3d: Optional[np.ndarray] = None  # [X, Y, Z] в метрах
    
    @property
    def shape(self) -> str:
        return self.class_name.split('_')[0]   # 'cup', 'cube', 'washer', 'lid'
    
    @property
    def color(self) -> str:
        parts = self.class_name.split('_')
        return '_'.join(parts[1:]) if len(parts) > 1 else ''
    
    @property
    def attributes(self) -> dict:
        """Признаки для политики LfD."""
        return {'shape': self.shape, 'color': self.color}

@dataclass
class PerceptionOutput:
    objects: list[DetectedObject]
    containers: dict   # {marker_id (int): corners_2d (np.ndarray), ...}
    container_positions_3d: dict  # {marker_id: np.ndarray [X,Y,Z]}
    timestamp: float
    rgb_frame: Optional[np.ndarray] = None   # для отладки