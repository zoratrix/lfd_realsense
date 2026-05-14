"""
generate_hand_marker.py — генерирует ArUco маркер руки для печати

Запусти один раз:
    python generate_hand_marker.py

Напечатай hand_marker_id10.png размером ~5×5 см.
Наклей на тыльную сторону ладони или на перчатку.
"""

import cv2
import numpy as np

HAND_MARKER_ID  = 10
MARKER_SIZE_PX  = 400   # размер маркера в пикселях
BORDER_PX       = 60    # белая рамка вокруг (нужна для детекции)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

marker = np.zeros((MARKER_SIZE_PX, MARKER_SIZE_PX), dtype=np.uint8)
marker = cv2.aruco.generateImageMarker(aruco_dict, HAND_MARKER_ID, MARKER_SIZE_PX, marker, 1)

total = MARKER_SIZE_PX + 2 * BORDER_PX
result = np.ones((total, total), dtype=np.uint8) * 255
result[BORDER_PX:BORDER_PX + MARKER_SIZE_PX,
       BORDER_PX:BORDER_PX + MARKER_SIZE_PX] = marker

# Подпись
cv2.putText(result, f"HAND MARKER  ID={HAND_MARKER_ID}  DICT_4X4_50",
            (10, total - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1)

out = "hand_marker_id10.png"
cv2.imwrite(out, result)
print(f"Сохранено: {out}")
print(f"Напечатай размером ~6×6 см (с рамкой) и наклей на тыльную сторону ладони.")
