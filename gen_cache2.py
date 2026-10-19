
# -*- coding: utf-8 -*-
"""
YOLOv8 Tomato-3 INT8 PTQ
Jetson Xavier / TensorRT 8.5.x

No torch / PyCUDA / onnxruntime.

One build produces:
  1. 1012_int8.engine
  2. 1012_int8_calib.cache
"""

import os
import glob
import time
import ctypes as ct

import cv2
import numpy as np
import tensorrt as trt


# ==========================================
# 1. Configuration
# ==========================================

ONNX_FILE = "1012fp32.onnx"
ENGINE_FILE = "1012_int8.engine"
CACHE_FILE = "1012_int8_calib.cache"

# Change to "train/images" if necessary
CALIB_DIR = "train"

MAX_CALIB_IMAGES = 100
IMG_SIZE = 640
WORKSPACE_GIB = 1


# ==========================================
# 2. CUDA Runtime
# ==========================================

cudart = ct.CDLL("libcudart.so")

cudart.cudaMalloc.argtypes = [
    ct.POINTER(ct.c_void_p),
    ct.c_size_t
]
cudart.cudaMalloc.restype = ct.c_int

cudart.cudaFree.argtypes = [
    ct.c_void_p
]
cudart.cudaFree.restype = ct.c_int

cudart.cudaMemcpy.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_size_t,
    ct.c_int
]
cudart.cudaMemcpy.restype = ct.c_int

cudart.cudaSetDevice.argtypes = [
    ct.c_int
]
cudart.cudaSetDevice.restype = ct.c_int


def cuda_check(code, operation):
    if code != 0:
        raise RuntimeError(
            "{} failed, CUDA error {}".format(
                operation, code
            )
        )


# ==========================================
# 3. Calibration preprocessing
# ==========================================

def preprocess_image(path):
    """
    Letterbox + BGR to RGB + Normalize
    Result: float32 NCHW [1,3,640,640]
    """

    img = cv2.imread(path)

    if img is None:
        raise RuntimeError(
            "Cannot read image: {}".format(path)
        )

    h, w = img.shape[:2]

    ratio = min(
        IMG_SIZE / h,
        IMG_SIZE / w
    )

    new_w = round(w * ratio)
    new_h = round(h * ratio)

    img = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    dw = IMG_SIZE - new_w
    dh = IMG_SIZE - new_h

    left = round(dw / 2 - 0.1)
    right = round(dw / 2 + 0.1)

    top = round(dh / 2 - 0.1)
    bottom = round(dh / 2 + 0.1)

    img = cv2.copyMakeBorder(
        img,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )

    if img.shape != (IMG_SIZE, IMG_SIZE, 3):
        raise RuntimeError(
            "Letterbox shape incorrect: {}".format(
                img.shape
            )
        )

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = img.astype(np.float32) / 255.0

    img = img.transpose(2, 0, 1)

    img = np.ascontiguousarray(
        img[None, :, :, :],
        dtype=np.float32
    )

    return img


# ==========================================
# 4. Native TensorRT entropy calibrator
# ==========================================

class YOLOEntropyCalibrator(
    trt.IInt8EntropyCalibrator2
):

    def __init__(
        self,
        img_dir,
        cache_file
    ):

        super().__init__()

        self.cache_file = cache_file
        self.batch_size = 1
        self.current_idx = 0

        self.img_paths = []

        for ext in (
            "*.jpg", "*.jpeg", "*.png",
            "*.JPG", "*.JPEG", "*.PNG"
        ):
            self.img_paths.extend(
                glob.glob(
                    os.path.join(img_dir, ext)
                )
            )

        self.img_paths = sorted(
            set(self.img_paths)
        )[:MAX_CALIB_IMAGES]

        if not self.img_paths:
            raise RuntimeError(
                "No calibration images in: {}".format(
                    img_dir
                )
            )

        self.allocation_size = (
            self.batch_size
            * 3
            * IMG_SIZE
            * IMG_SIZE
            * np.dtype(np.float32).itemsize
        )

        self.device_mem = ct.c_void_p()

        print(
            "Calibration images:",
            len(self.img_paths),
            flush=True
        )

        cuda_check(
            cudart.cudaMalloc(
                ct.byref(self.device_mem),
                self.allocation_size
            ),
            "cudaMalloc"
        )

        if not self.device_mem.value:
            raise RuntimeError(
                "CUDA allocation returned null pointer"
            )

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):

        if self.current_idx >= len(self.img_paths):
            return None

        path = self.img_paths[self.current_idx]

        img = preprocess_image(path)

        if img.nbytes != self.allocation_size:
            raise RuntimeError(
                "Calibration input byte size mismatch"
            )

        cuda_check(
            cudart.cudaMemcpy(
                self.device_mem,
                ct.c_void_p(img.ctypes.data),
                self.allocation_size,
                1  # cudaMemcpyHostToDevice
            ),
            "cudaMemcpy H2D"
        )

        self.current_idx += 1

        if (
            self.current_idx == 1
            or self.current_idx % 10 == 0
            or self.current_idx == len(self.img_paths)
        ):
            print(
                "Calibration input {}/{}: {}".format(
                    self.current_idx,
                    len(self.img_paths),
                    os.path.basename(path)
                ),
                flush=True
            )

        return [int(self.device_mem.value)]

    def read_calibration_cache(self):
        # Force fresh calibration for this model.
        return None

    def write_calibration_cache(self, cache):

        with open(self.cache_file, "wb") as f:
            f.write(cache)

        print(
            "Calibration cache written:",
            self.cache_file,
            flush=True
        )

    def free(self):

        if self.device_mem.value:

            code = cudart.cudaFree(
                self.device_mem
            )

            self.device_mem = ct.c_void_p()

            cuda_check(
                code,
                "cudaFree"
            )


# ==========================================
# 5. Build and save INT8 Engine
# ==========================================

def main():

    print("=" * 60)
    print("YOLOv8 Tomato-3 INT8 Calibration")
    print("TensorRT:", trt.__version__)
    print("=" * 60, flush=True)

    if not os.path.isfile(ONNX_FILE):
        raise FileNotFoundError(
            "Missing ONNX: {}".format(ONNX_FILE)
        )

    if not os.path.isdir(CALIB_DIR):
        raise NotADirectoryError(
            "Missing calibration dir: {}".format(
                CALIB_DIR
            )
        )

    # Never overwrite an existing benchmark model.
    for path in (ENGINE_FILE, CACHE_FILE):
        if os.path.exists(path):
            raise RuntimeError(
                "{} already exists. "
                "Rename or archive it first."
                .format(path)
            )

    cuda_check(
        cudart.cudaSetDevice(0),
        "cudaSetDevice"
    )

    logger = trt.Logger(trt.Logger.INFO)

    builder = trt.Builder(logger)

    network = builder.create_network(
        1 << int(
            trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
        )
    )

    parser = trt.OnnxParser(
        network, logger
    )

    print("Parsing ONNX...", flush=True)

    with open(ONNX_FILE, "rb") as f:
        parsed = parser.parse(f.read())

    if not parsed:

        for i in range(parser.num_errors):
            print(
                parser.get_error(i),
                flush=True
            )

        raise RuntimeError(
            "ONNX parsing failed"
        )

    if (
        network.num_inputs != 1
        or network.num_outputs != 1
    ):
        raise RuntimeError(
            "Expected one input and one output. "
            "Inputs={}, Outputs={}".format(
                network.num_inputs,
                network.num_outputs
            )
        )

    input_tensor = network.get_input(0)
    output_tensor = network.get_output(0)

    input_shape = tuple(input_tensor.shape)
    output_shape = tuple(output_tensor.shape)

    print("Input:", input_tensor.name, input_shape)
    print("Output:", output_tensor.name, output_shape)
    print("Input dtype:", input_tensor.dtype)

    if input_shape != (1, 3, 640, 640):
        raise RuntimeError(
            "Unexpected input shape: {}".format(
                input_shape
            )
        )

    if output_shape not in (
        (1, 7, 8400),
        (1, 8400, 7)
    ):
        raise RuntimeError(
            "Not a standard 3-class YOLOv8 "
            "detection output: {}".format(
                output_shape
            )
        )

    if input_tensor.dtype != trt.float32:
        raise RuntimeError(
            "Expected FP32 ONNX input"
        )

    print(
        "PASS: Three-class ONNX structure",
        flush=True
    )

    config = builder.create_builder_config()

    config.set_flag(
        trt.BuilderFlag.INT8
    )

    config.set_flag(
        trt.BuilderFlag.FP16
    )

    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        WORKSPACE_GIB << 30
    )

    print(
        "Precision flags: INT8 + FP16",
        flush=True
    )

    print(
        "Workspace: {} GiB".format(
            WORKSPACE_GIB
        ),
        flush=True
    )

    calibrator = None
    started = time.time()

    try:

        calibrator = YOLOEntropyCalibrator(
            CALIB_DIR,
            CACHE_FILE
        )

        config.int8_calibrator = calibrator

        print(
            "Starting TensorRT calibration and build...",
            flush=True
        )

        serialized = builder.build_serialized_network(
            network,
            config
        )

        if serialized is None:
            raise RuntimeError(
                "TensorRT engine build returned None"
            )

        # Write engine only after the build succeeds.
        temp_file = ENGINE_FILE + ".tmp"

        try:
            with open(temp_file, "wb") as f:
                f.write(bytes(serialized))

            if os.path.getsize(temp_file) == 0:
                raise RuntimeError(
                    "Generated engine is empty"
                )

            os.replace(
                temp_file,
                ENGINE_FILE
            )

        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        print(
            "Engine saved:",
            ENGINE_FILE,
            flush=True
        )

        if not os.path.isfile(CACHE_FILE):
            print(
                "WARNING: Calibration cache "
                "was not written.",
                flush=True
            )
        else:
            print(
                "Cache size:",
                os.path.getsize(CACHE_FILE),
                "bytes",
                flush=True
            )

    finally:

        if calibrator is not None:
            calibrator.free()

    elapsed = time.time() - started

    print("=" * 60)
    print("BUILD COMPLETED")
    print("Engine:", ENGINE_FILE)
    print("Cache:", CACHE_FILE)
    print(
        "Elapsed: {:.1f} seconds".format(
            elapsed
        )
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
