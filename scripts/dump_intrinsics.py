"""
dump_intrinsics.py — снимает параметры камеры для текущей конфигурации потока.

Запуск:
    python scripts/dump_intrinsics.py                    # печатает и пишет config
    python scripts/dump_intrinsics.py --demos data/demos # + кладёт intrinsics.json
                                                         #   в каждую папку демо
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyrealsense2 as rs

STREAM_W, STREAM_H, STREAM_FPS = 640, 480, 30


def get_intrinsics() -> dict:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, STREAM_W, STREAM_H, rs.format.bgr8, STREAM_FPS)
    config.enable_stream(rs.stream.depth, STREAM_W, STREAM_H, rs.format.z16, STREAM_FPS)
    profile = pipeline.start(config)
    try:
        # Кадры выравниваются на color -> нужны intrinsics ИМЕННО color-потока
        color_stream = profile.get_stream(rs.stream.color)
        intr = color_stream.as_video_stream_profile().get_intrinsics()
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        return {
            "fx": float(intr.fx), "fy": float(intr.fy),
            "ppx": float(intr.ppx), "ppy": float(intr.ppy),
            "width": int(intr.width), "height": int(intr.height),
            "depth_scale": float(depth_scale),
            "coeffs": [float(c) for c in intr.coeffs],
            "model": str(intr.model),
        }
    finally:
        pipeline.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default=None,
                    help="корень с демонстрациями, напр. data/demos")
    args = ap.parse_args()

    d = get_intrinsics()

    print("\nПараметры камеры (color-поток, 640x480):")
    for k in ("fx", "fy", "ppx", "ppy", "depth_scale"):
        print(f"  {k:12s} = {d[k]}")

    print("\nВставьте это в config/perception.yaml:\n")
    print("camera:")
    print(f"  fx: {d['fx']}")
    print(f"  fy: {d['fy']}")
    print(f"  cx: {d['ppx']}")
    print(f"  cy: {d['ppy']}")
    print(f"  depth_scale: {d['depth_scale']}")

    Path("config").mkdir(exist_ok=True)
    Path("config/intrinsics.json").write_text(
        json.dumps(d, indent=2), encoding="utf-8")
    print("\nСохранено: config/intrinsics.json")

    if args.demos:
        root = Path(args.demos)
        n = 0
        for demo_dir in sorted(root.iterdir()):
            if not (demo_dir / "rgb").is_dir():
                continue
            (demo_dir / "intrinsics.json").write_text(
                json.dumps(d, indent=2), encoding="utf-8")
            n += 1
            print(f"  intrinsics.json -> {demo_dir}")
        print(f"\nЗаписано в {n} папок демонстраций")