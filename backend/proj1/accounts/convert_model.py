#!/usr/bin/env python3
"""
Run this script during the Render build step (with TF+tf2onnx installed).
It converts best_model (1).h5 → model.onnx so the runtime only needs onnxruntime.
"""
import os
import json
import h5py

model_dir = os.path.dirname(os.path.abspath(__file__))
h5_path = os.path.join(model_dir, 'best_model (1).h5')
onnx_path = os.path.join(model_dir, 'model.onnx')

if os.path.exists(onnx_path):
    print(f"ONNX model already exists at {onnx_path}, skipping conversion.")
else:
    print(f"Patching H5 config before loading...")
    with h5py.File(h5_path, 'r+') as f:
        if 'model_config' in f.attrs:
            raw = f.attrs['model_config']
            if isinstance(raw, bytes):
                raw = raw.decode('utf-8')
            cfg = json.loads(raw)

            def clean(c):
                if isinstance(c, dict):
                    c.pop('quantization_config', None)
                    for v in list(c.values()):
                        clean(v)
                elif isinstance(c, list):
                    for i in c:
                        clean(i)

            clean(cfg)
            f.attrs['model_config'] = json.dumps(cfg).encode('utf-8')
            print("H5 config patched.")

    print("Loading Keras model...")
    import tensorflow as tf
    import tf2onnx

    class PatchedInputLayer(tf.keras.layers.InputLayer):
        def __init__(self, *args, **kwargs):
            kwargs.pop('optional', None)
            kwargs.pop('ragged', None)
            if 'batch_shape' in kwargs:
                kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
            super().__init__(*args, **kwargs)

    model = tf.keras.models.load_model(
        h5_path,
        custom_objects={'InputLayer': PatchedInputLayer},
        compile=False
    )
    print(f"Model loaded. Input names: {[i.name for i in model.inputs]}")

    print("Converting to ONNX (opset 13)...")
    input_sig = [tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name='input')]
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_sig,
        opset=13,
        output_path=onnx_path
    )
    print(f"ONNX model saved to {onnx_path} ✓")
