import time
import numpy as np
import pyrealsense2 as rs
from .detector import ObjectDetector
from .aruco_tracker import ArucoTracker
from .postprocessor import PostProcessor
from .types import PerceptionOutput

class FrameProcessor:
    """
    Главный класс модуля распознавания.
    Использование:
        fp = FrameProcessor.from_config('config/perception.yaml')
        fp.initialize_containers()   # один раз при старте
        output = fp.process_frame()  # в основном цикле
    """
    
    def __init__(self, detector: ObjectDetector,
                 aruco: ArucoTracker,
                 postprocessor: PostProcessor,
                 pipeline: rs.pipeline,
                 align: rs.align):
        self.detector = detector
        self.aruco = aruco
        self.postprocessor = postprocessor
        self.pipeline = pipeline
        self.align = align
        self._containers_initialized = False

    @classmethod
    def from_config(cls, config_path: str) -> 'FrameProcessor':
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        
        # Настройка RealSense
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = pipeline.start(config)
        
        # depth scale (обычно 0.001 для D4xx — 1 единица = 1 мм)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        
        # Camera intrinsics из профиля
        color_stream = profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        
        # Align depth to color
        align = rs.align(rs.stream.color)
        
        detector = ObjectDetector(
            weights_path=cfg['model']['weights'],
            conf_thresh=cfg['model'].get('conf_thresh', 0.4),
            device=cfg['model'].get('device', 'auto')
        )
        aruco = ArucoTracker(
            marker_size_m=cfg['aruco'].get('marker_size_m', 0.04)
        )
        postprocessor = PostProcessor(
            depth_scale=depth_scale,
            camera_intrinsics=intrinsics
        )
        
        return cls(detector, aruco, postprocessor, pipeline, align)

    def initialize_containers(self, n_warmup: int = 30) -> dict:
        """
        Прогревает камеру и детектирует контейнеры один раз.
        Вызвать при старте сессии перед основным циклом.
        """
        print(f"[FrameProcessor] Прогрев камеры ({n_warmup} кадров)...")
        for _ in range(n_warmup):
            self.pipeline.wait_for_frames()
        
        # Делаем несколько попыток — ArUco может не найти с первого кадра
        for attempt in range(5):
            frames = self._get_aligned_frames()
            rgb = frames['rgb']
            depth = frames['depth']
            intr = frames['intrinsics']
            
            # Обновляем intrinsics в постпроцессоре (на случай if from_config)
            self.postprocessor.intrinsics = intr
            
            cache, cache_3d = self.aruco.detect_and_cache(
                rgb, depth, self.postprocessor.depth_scale, intr
            )
            
            if cache:
                print(f"[FrameProcessor] Контейнеры найдены: id={list(cache.keys())}")
                self._containers_initialized = True
                return cache_3d
            
            print(f"[FrameProcessor] Попытка {attempt+1}/5: маркеры не найдены...")
        
        print("[FrameProcessor] WARN: контейнеры не найдены, продолжаем без них")
        return {}

    def process_frame(self, return_rgb: bool = False) -> PerceptionOutput:
        """Основной метод — вызывать в цикле."""
        t0 = time.monotonic()
        
        frames = self._get_aligned_frames()
        rgb = frames['rgb']
        depth = frames['depth']
        
        # 1. YOLOv5 детекция
        raw_detections = self.detector.predict(rgb)
        
        # 2. Постпроцессинг: 3D + верификация
        objects = self.postprocessor.process(raw_detections, depth)
        
        # 3. Берём кэшированные контейнеры
        containers, containers_3d = self.aruco.get_cached()
        
        elapsed = time.monotonic() - t0
        # print(f"[FrameProcessor] frame time: {elapsed*1000:.1f} мс")
        
        return PerceptionOutput(
            objects=objects,
            containers=containers,
            container_positions_3d=containers_3d,
            timestamp=time.time(),
            rgb_frame=rgb if return_rgb else None
        )

    def _get_aligned_frames(self) -> dict:
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        
        rgb = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        intr = color_frame.get_profile().as_video_stream_profile().get_intrinsics()
        
        return {'rgb': rgb, 'depth': depth, 'intrinsics': intr}

    def stop(self):
        self.pipeline.stop()