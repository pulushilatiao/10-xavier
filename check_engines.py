
from pathlib import Path
import tensorrt as trt

root = Path(__file__).resolve().parent

models = {
    "FP16": root / "1012_fp16.engine",
    #"INT8": root / "yolov8n_jetson_int8-100916PM.engine"
}

logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)

print("TensorRT:", trt.__version__)

for name, path in models.items():
    print("\n==========", name, "==========")

    if not path.exists():
        print("File not found:", path)
        continue

    print("Model:", path.name)
    print("Size MB:", round(
        path.stat().st_size / 1024**2, 2
    ))

    engine = runtime.deserialize_cuda_engine(
        path.read_bytes()
    )

    if engine is None:
        print("Engine loading FAILED")
        continue

    print("Bindings:", engine.num_bindings)

    for i in range(engine.num_bindings):
        print(
            "Name:", engine.get_binding_name(i),
            "| Input:", engine.binding_is_input(i),
            "| Shape:", tuple(
                engine.get_binding_shape(i)
            ),
            "| Dtype:", engine.get_binding_dtype(i)
        )

    del engine

print("\nInspection completed.")
