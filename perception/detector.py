import torch
import numpy as np
import cv2
from pathlib import Path

class ObjectDetector:
    """YOLOv5s inference обёртка."""
    
    # CLASS_NAMES = [
    #     'cup_silver', 'cup_dark_grey',
    #     'cube_blue', 'cube_red', 'cube_green',
    #     'washer_green', 'lid', 'aruco_container'
    # ]

    
    
    def __init__(self, weights_path: str,
                 conf_thresh: float = 0.4,
                 iou_thresh: float = 0.45,
                 device: str = 'auto'):
        

        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Загружаем через ultralytics (поддерживает YOLOv5 и v8)
        from ultralytics import YOLO
        self.model = YOLO(weights_path)
        self.model.to(device)
        
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        self.class_names = self.model.names  # dict {0: 'aruco_container', ...}
        print(f"[Detector] Загружено: {weights_path}, device={device}")


    def predict(self, rgb_frame: np.ndarray) -> list[dict]:
        """
        Возвращает список детекций:
        [{'class_name', 'confidence', 'bbox': (x,y,w,h), 'mask': None}, ...]
        """
        results = self.model.predict(
            rgb_frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False
        )[0]
        
        detections = []
        boxes = results.boxes
        if boxes is None:
            return detections
        
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
            
            det = {
                'class_name': self.class_names[cls_id],
                'confidence': conf,
                'bbox': (x, y, w, h),
                'mask': None
            }
            
            # Если YOLOv5-seg — достаём маску
            if results.masks is not None:
                mask_data = results.masks.data[i].cpu().numpy()
                # Ресайз маски до размера кадра
                mask_resized = cv2.resize(
                    mask_data,
                    (rgb_frame.shape[1], rgb_frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                det['mask'] = mask_resized.astype(bool)
            
            detections.append(det)
        
        return detections