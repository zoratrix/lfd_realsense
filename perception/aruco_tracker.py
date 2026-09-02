import cv2
import numpy as np
import logging

from .geometry import CameraIntrinsics, pixel_to_3d

logger = logging.getLogger(__name__)

class ArucoTracker:
    """
    Детектирует ArUco-маркеры на контейнерах.
    Позиции кэшируются — контейнеры считаются стационарными.
    """
    
    DICT_TYPE = cv2.aruco.DICT_4X4_50

    def __init__(self, marker_size_m: float = 0.04):
        self.marker_size_m = marker_size_m
        aruco_dict = cv2.aruco.getPredefinedDictionary(self.DICT_TYPE)
        params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        
        # Кэш: {marker_id: corners_2d}
        self._cache: dict[int, np.ndarray] = {}
        self._cache_3d: dict[int, np.ndarray] = {}
        self._initialized = False

    def detect_and_cache(self, rgb_frame: np.ndarray,
                          depth_frame: np.ndarray,
                          depth_scale: float,
                          camera_intrinsics) -> tuple[dict, dict]:
        """
        Запускать один раз при старте сессии.
        Возвращает ({marker_id: corners_2d}, {marker_id: pos_3d}).
        """
        corners, ids, _ = self.detector.detectMarkers(rgb_frame)
        
        if ids is None:
            logger.warning("ArUco: маркеры не найдены!")
            return self._cache, self._cache_3d
        
        for corner, mid in zip(corners, ids.flatten()):
            self._cache[int(mid)] = corner[0]  # shape (4, 2)
            # Центр маркера в пикселях
            cx = int(corner[0][:, 0].mean())
            cy = int(corner[0][:, 1].mean())
            pos3d = self._pixel_to_3d(cx, cy, depth_frame,
                                       depth_scale, camera_intrinsics)
            if pos3d is not None:
                self._cache_3d[int(mid)] = pos3d
                logger.info(f"ArUco id={mid}: 3D={pos3d.round(3)} м")
        
        self._initialized = True
        return self._cache, self._cache_3d

    def get_cached(self) -> tuple[dict, dict]:
        return self._cache, self._cache_3d

    @staticmethod
    def _pixel_to_3d(cx: int, cy: int,
                      depth_frame: np.ndarray,
                      depth_scale: float,
                      intrinsics) -> np.ndarray | None:
        """Обратная проекция центра маркера в 3D через карту глубины."""
        intr = CameraIntrinsics.from_rs(intrinsics, depth_scale)
        # Маркер плоский и малого размера — фиксированный патч 5x5 (radius=2)
        return pixel_to_3d(cx, cy, depth_frame, intr,
                           radius=2, min_valid=1)
    
    def draw_debug(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        if self._cache:
            corners_list = [c[np.newaxis] for c in self._cache.values()]
            ids_arr = np.array(list(self._cache.keys()))
            cv2.aruco.drawDetectedMarkers(out, corners_list, ids_arr)
        return out