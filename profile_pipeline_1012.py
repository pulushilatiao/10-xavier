
# -*- coding: utf-8 -*-
"""
YOLOv8 FP16 / INT8 pipeline profiler
Jetson Xavier / TensorRT 8.5.x

Reuses eval_precision_1012TEST.py
No PyTorch / Ultralytics / PyCUDA.

Measures:
  image read
  preprocessing
  TensorRT infer() wall time
  TensorRT GPU event time
  decode / score filtering
  class-aware NMS
  end-to-end latency

Does NOT alter models or inference thresholds.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

import eval_precision_1012TEST as base


ROOT = Path(__file__).resolve().parent

MODELS = {
    "FP16": ROOT / "1012_fp16.engine",
    "INT8": ROOT / "1012_int8.engine",
}

CONF = 0.001
NMS_IOU = 0.70
MAX_DET = 300

EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def get_images():
    folder = ROOT / "val" / "images"

    if not folder.is_dir():
        folder = ROOT / "val"

    if not folder.is_dir():
        raise RuntimeError(
            "Validation folder not found: {}".format(folder)
        )

    paths = sorted(
        p for p in folder.rglob("*")
        if (
            p.is_file()
            and p.suffix.lower() in EXTS
            and "labels" not in p.parts
        )
    )

    if not paths:
        raise RuntimeError("No validation images found")

    return paths


def profiled_postprocess(
    raw, width, height,
    ratio, left, top, nc
):
    """
    Same detection processing as the existing script,
    but with separate decode and NMS timings.

    No pre-NMS Top-K is added.
    """

    t0 = time.perf_counter()

    a = raw[0].astype(np.float32)

    if a.shape[1] == 8400:
        a = a.T

    if a.shape != (8400, 4 + nc):
        raise RuntimeError(
            "Unexpected YOLO output: {}".format(a.shape)
        )

    if not np.all(np.isfinite(a)):
        raise RuntimeError("NaN or Inf in prediction")

    scores = a[:, 4:].max(axis=1)
    cls = a[:, 4:].argmax(axis=1)

    valid = scores > CONF

    a = a[valid]
    scores = scores[valid]
    cls = cls[valid]

    candidate_count = len(a)

    if not candidate_count:
        t1 = time.perf_counter()
        return np.empty((0, 6), np.float32), {
            "decode_ms": (t1-t0)*1000,
            "nms_ms": 0.0,
            "restore_ms": 0.0,
            "candidates": 0,
            "after_nms": 0,
            "kept": 0,
            "capped": False,
        }

    boxes = np.empty((len(a), 4), np.float32)

    boxes[:, :2] = (
        a[:, :2] - a[:, 2:4] / 2
    )

    boxes[:, 2:] = (
        a[:, :2] + a[:, 2:4] / 2
    )

    t1 = time.perf_counter()

    # ======================================
    # Existing class-aware OpenCV NMS
    # ======================================

    picks = []

    for c in np.unique(cls):
        idx = np.flatnonzero(cls == c)

        xywh = np.column_stack(
            (
                boxes[idx, :2],
                np.maximum(
                    0,
                    boxes[idx, 2:] - boxes[idx, :2]
                )
            )
        )

        keep = cv2.dnn.NMSBoxes(
            xywh.tolist(),
            scores[idx].tolist(),
            CONF,
            NMS_IOU
        )

        if keep is not None and len(keep):
            picks.extend(
                idx[np.asarray(keep).reshape(-1)].tolist()
            )

    after_nms = len(picks)

    t2 = time.perf_counter()

    if not picks:
        return np.empty((0, 6), np.float32), {
            "decode_ms": (t1-t0)*1000,
            "nms_ms": (t2-t1)*1000,
            "restore_ms": 0.0,
            "candidates": candidate_count,
            "after_nms": 0,
            "kept": 0,
            "capped": False,
        }

    picks = np.asarray(picks, dtype=int)

    picks = picks[
        np.argsort(-scores[picks])[:MAX_DET]
    ]

    capped = after_nms > MAX_DET

    b = boxes[picks].copy()

    b[:, [0, 2]] = np.clip(
        (b[:, [0, 2]] - left) / ratio,
        0,
        width
    )

    b[:, [1, 3]] = np.clip(
        (b[:, [1, 3]] - top) / ratio,
        0,
        height
    )

    pred = np.column_stack(
        (
            b,
            scores[picks],
            cls[picks]
        )
    ).astype(np.float32)

    t3 = time.perf_counter()

    return pred, {
        "decode_ms": (t1-t0)*1000,
        "nms_ms": (t2-t1)*1000,
        "restore_ms": (t3-t2)*1000,
        "candidates": candidate_count,
        "after_nms": after_nms,
        "kept": len(pred),
        "capped": capped,
    }


def describe(values):
    a = np.asarray(values, dtype=np.float64)

    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def run_model(name, path, images, repeats, cuda):

    print("\n" + "=" * 60)
    print("PROFILING:", name)
    print("Engine:", path)
    print("=" * 60, flush=True)

    model = base.Engine(path, cuda)

    try:
        if model.nc != 3:
            raise RuntimeError(
                "Expected 3 classes, got {}".format(model.nc)
            )

        first = cv2.imread(str(images[0]))

        if first is None:
            raise RuntimeError("Cannot read first image")

        warm, _, _, _ = base.preprocess(first)

        print("Warmup: 10 iterations", flush=True)

        for _ in range(10):
            model.infer(warm)

        records = []
        reference_checked = False

        for repeat_idx in range(repeats):

            print(
                "Repeat {}/{}".format(
                    repeat_idx + 1, repeats
                ),
                flush=True
            )

            for file in images:

                e2e_start = time.perf_counter()

                t = time.perf_counter()
                img = cv2.imread(str(file))
                read_ms = (time.perf_counter()-t)*1000

                if img is None:
                    raise RuntimeError(
                        "Cannot read {}".format(file)
                    )

                h, w = img.shape[:2]

                t = time.perf_counter()
                inp, ratio, left, top = base.preprocess(img)
                preprocess_ms = (
                    time.perf_counter()-t
                )*1000

                t = time.perf_counter()
                output, gpu_ms, _ = model.infer(inp)
                infer_wall_ms = (
                    time.perf_counter()-t
                )*1000

                post_start = time.perf_counter()

                pred, detail = profiled_postprocess(
                    output, w, h,
                    ratio, left, top, model.nc
                )

                post_wall_ms = (
                    time.perf_counter()-post_start
                )*1000

                e2e_ms = (
                    time.perf_counter()-e2e_start
                )*1000

                # Ensure this instrumented implementation
                # agrees with the original postprocess.
                if not reference_checked:

                    original = base.postprocess(
                        output, (w, h),
                        ratio, left, top, model.nc
                    )

                    if (
                        original.shape != pred.shape
                        or not np.allclose(
                            original, pred,
                            rtol=1e-5,
                            atol=1e-4
                        )
                    ):
                        raise RuntimeError(
                            "Profiler postprocess differs "
                            "from the original evaluator."
                        )

                    reference_checked = True
                    print(
                        "PASS: Postprocess matches original",
                        flush=True
                    )

                record = {
                    "file": file.name,
                    "repeat": repeat_idx + 1,
                    "read_ms": read_ms,
                    "preprocess_ms": preprocess_ms,
                    "infer_wall_ms": infer_wall_ms,
                    "gpu_ms": gpu_ms,
                    "post_ms": post_wall_ms,
                    "e2e_ms": e2e_ms,
                }

                record.update(detail)
                records.append(record)

        keys = [
            "read_ms",
            "preprocess_ms",
            "infer_wall_ms",
            "gpu_ms",
            "decode_ms",
            "nms_ms",
            "restore_ms",
            "post_ms",
            "e2e_ms",
        ]

        stats = {
            k: describe([r[k] for r in records])
            for k in keys
        }

        print("\nSTAGE LATENCY (ms)")
        print("{:<20} {:>10} {:>10}".format(
            "Stage", "Mean", "P95"
        ))
        print("-" * 44)

        for k in keys:
            print(
                "{:<20} {:>10.3f} {:>10.3f}".format(
                    k,
                    stats[k]["mean"],
                    stats[k]["p95"]
                )
            )

        avg_e2e = stats["e2e_ms"]["mean"]

        print("\nBOTTLENECK SHARES")
        print(
            "Image read: {:.1f}%".format(
                100 * stats["read_ms"]["mean"]/avg_e2e
            )
        )
        print(
            "Preprocessing: {:.1f}%".format(
                100 * stats["preprocess_ms"]["mean"]/avg_e2e
            )
        )
        print(
            "Infer wall: {:.1f}%".format(
                100 * stats["infer_wall_ms"]["mean"]/avg_e2e
            )
        )
        print(
            "Postprocess: {:.1f}%".format(
                100 * stats["post_ms"]["mean"]/avg_e2e
            )
        )
        print(
            "NMS alone: {:.1f}%".format(
                100 * stats["nms_ms"]["mean"]/avg_e2e
            )
        )

        capped_count = sum(r["capped"] for r in records)

        print("\nCANDIDATES")
        print(
            "Mean before NMS: {:.1f}".format(
                np.mean([r["candidates"] for r in records])
            )
        )
        print(
            "Mean after NMS: {:.1f}".format(
                np.mean([r["after_nms"] for r in records])
            )
        )
        print(
            "Capped at MAX_DET: {}/{}".format(
                capped_count,
                len(records)
            )
        )
        print(
            "End-to-end FPS: {:.3f}".format(
                1000 / avg_e2e
            )
        )

        return {
            "stats": stats,
            "records": records,
            "capped_count": capped_count,
            "samples": len(images),
            "repeats": repeats,
        }

    finally:
        model.close()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repeats",
        type=int,
        default=3
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=0
    )

    args = parser.parse_args()

    if args.repeats < 1 or args.max_images < 0:
        parser.error("Invalid test parameters")

    images = get_images()

    if args.max_images:
        images = images[:args.max_images]

    print("Images:", len(images))
    print("Repeats:", args.repeats)

    for path in MODELS.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    cuda = base.Cuda()

    result = {}

    for name, path in MODELS.items():
        result[name] = run_model(
            name, path, images,
            args.repeats, cuda
        )

    output_path = ROOT / "pipeline_profile_1012.json"

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
