#!/usr/bin/env python3
"""
Convert best_model.h5 to ONNX format during the Render build step.
Run with TensorFlow and tf2onnx installed. Exits with code 1 on any failure.
"""
import os
import sys
import json
import traceback

model_dir = os.path.dirname(os.path.abspath(__file__))
h5_path = os.path.join(model_dir, 'best_model.h5')
onnx_path = os.path.join(model_dir, 'model.onnx')

print(f"=== ONNX Conversion Script ===")
print(f"H5  path : {h5_path}")
print(f"ONNX path: {onnx_path}")
print(f"H5 exists: {os.path.exists(h5_path)}")

if not os.path.exists(h5_path):
    print(f"ERROR: H5 model not found!")
    sys.exit(1)

# Always regenerate so stale files never hide issues
if os.path.exists(onnx_path):
    print("Removing existing model.onnx to regenerate...")
    os.remove(onnx_path)

try:
    import h5py
    import json as _json

    print("Patching H5 config (removing Keras 3 incompatible fields)...")
    with h5py.File(h5_path, 'r+') as f:
        if 'model_config' in f.attrs:
            raw = f.attrs['model_config']
            if isinstance(raw, bytes):
                raw = raw.decode('utf-8')
            cfg = _json.loads(raw)

            def clean(c):
                if isinstance(c, dict):
                    c.pop('quantization_config', None)
                    for v in list(c.values()):
                        clean(v)
                elif isinstance(c, list):
                    for i in c:
                        clean(i)

            clean(cfg)
            f.attrs['model_config'] = _json.dumps(cfg).encode('utf-8')
            print("H5 config patched OK.")

    print("Importing TensorFlow...")
    import tensorflow as tf
    print(f"TF version: {tf.__version__}  Keras: {tf.keras.__version__}")

    class PatchedInputLayer(tf.keras.layers.InputLayer):
        def __init__(self, *args, **kwargs):
            kwargs.pop('optional', None)
            kwargs.pop('ragged', None)
            if 'batch_shape' in kwargs:
                kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
            super().__init__(*args, **kwargs)

    print("Loading Keras model (compile=False)...")
    model = tf.keras.models.load_model(
        h5_path,
        custom_objects={'InputLayer': PatchedInputLayer},
        compile=False
    )
    print("Model loaded!")
    print(f"  inputs : {[inp.name for inp in model.inputs]}")
    print(f"  shapes : {[list(inp.shape) for inp in model.inputs]}")
    print(f"  outputs: {[out.name for out in model.outputs]}")

    import tf2onnx
    print(f"tf2onnx version: {tf2onnx.__version__}")

    # Use the model's actual input shape (replace batch dim with None)
    input_shape = list(model.inputs[0].shape)
    input_shape[0] = None
    print(f"Converting with input shape {input_shape} and opset=13...")

    input_sig = [tf.TensorSpec(shape=input_shape, dtype=tf.float32, name='input')]
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_sig,
        opset=13,
        output_path=onnx_path
    )

    if not os.path.exists(onnx_path):
        print("ERROR: model.onnx was not written by tf2onnx!")
        sys.exit(1)

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"model.onnx saved ({size_mb:.1f} MB) ✓")

except Exception:
    print("\n=== CONVERSION FAILED ===")
    traceback.print_exc()
    sys.exit(1)
