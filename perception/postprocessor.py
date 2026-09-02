import numpy as np
import cv2
from .types import DetectedObject
from .geometry import CameraIntrinsics, median_depth_raw, deproject

class PostProcessor:
    """
    1. Деprojectирует bbox/маску в 3D через depth.
    2. Верифицирует washer vs cube_green по circularity
       при низком confidence.
    """
    
    CIRCULARITY_THRESHOLD = 0.78   # washer → ~1.0, cube → ~0.65-0.75
    LOW_CONF_THRESHOLD = 0.55       # ниже → запускаем геометрическую верификацию

    def __init__(self, depth_scale: float, camera_intrinsics):
        self.depth_scale = depth_scale
        self.intrinsics = camera_intrinsics

    def process(self, detections: list[dict],
                depth_frame: np.ndarray) -> list[DetectedObject]:
        objects = []
        for det in detections:
            class_name = det['class_name']
            conf = det['confidence']
            bbox = det['bbox']
            mask = det['mask']
            
            # Геометрическая верификация washer/cube_green при низком conf
            if (class_name in ('washer_green', 'cube_green')
                    and conf < self.LOW_CONF_THRESHOLD):
                class_name = self._verify_washer_or_cube(bbox, mask, class_name)
            
            # 3D центр объекта
            center_3d = self._get_center_3d(bbox, mask, depth_frame)
            
            objects.append(DetectedObject(
                class_name=class_name,
                confidence=conf,
                bbox=bbox,
                mask=mask,
                center_3d=center_3d
            ))
        
        return objects

    def _verify_washer_or_cube(self, bbox, mask, original_class: str) -> str:
        """Circularity = 4π·Area / Perimeter² ≈ 1 для круга, < 0.8 для квадрата."""
        x, y, w, h = bbox
        
        if mask is not None:
            # Маска есть — считаем circularity по реальному контуру
            mask_u8 = mask[y:y+h, x:x+w].astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
        else:
            # Маски нет — делаем эллипс внутри bbox как приближение
            # Синтетический контур эллипса в bbox
            ellipse_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(ellipse_mask, (w//2, h//2), (w//2-2, h//2-2),
                        0, 0, 360, 255, -1)
            contours, _ = cv2.findContours(ellipse_mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return original_class
        
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        if perimeter < 1:
            return original_class
        
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        if circularity >= self.CIRCULARITY_THRESHOLD:
            return 'washer_green'
        else:
            return 'cube_green'

    def _get_center_3d(self, bbox, mask,
                        depth_frame: np.ndarray) -> np.ndarray | None:
        """Берёт медианную глубину по маске или патчу в центре bbox."""
        x, y, w, h = bbox
        cx_px = x + w // 2
        cy_px = y + h // 2

        intr = CameraIntrinsics.from_rs(self.intrinsics, self.depth_scale)

        if mask is not None:
            # Глубина по всем валидным пикселям маски объекта
            depth_values = depth_frame[mask & (depth_frame > 0)]
            if len(depth_values) < 5:
                return None
            depth_raw = float(np.median(depth_values))
        else:
            # Маски нет — патч в центре bbox, радиус пропорционален размеру
            radius = max(5, min(w, h) // 5)
            depth_raw = median_depth_raw(depth_frame, cx_px, cy_px,
                                         radius=radius, min_valid=5)
            if depth_raw is None:
                return None

        return deproject(cx_px, cy_px, depth_raw, intr)