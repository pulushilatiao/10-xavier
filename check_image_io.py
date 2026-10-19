from pathlib import Path
import cv2
import time

root = Path("val/images")
if not root.exists():
    root = Path("val")

names = [
    "IMG_7314.jpg",
    "IMG_7317.jpg",
    "IMG_7319.jpg",
    "IMG_7328.jpg",
    "IMG_7419.jpg",
]

for name in names:
    p = root / name
    if not p.exists():
        print(name, "not found")
        continue

    times = []
    shape = None

    for _ in range(5):
        t0 = time.perf_counter()
        img = cv2.imread(str(p))
        times.append(
            (time.perf_counter() - t0) * 1000
        )
        shape = img.shape if img is not None else None

    print(
        name,
        "size_MB=", round(p.stat().st_size / 1024**2, 2),
        "shape=", shape,
        "read_ms=", [round(t, 1) for t in times]
    )