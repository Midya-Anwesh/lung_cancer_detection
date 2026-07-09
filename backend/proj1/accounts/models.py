from django.db import models
from django.core.files.base import ContentFile

import os
import logging
from io import BytesIO
from threading import Lock

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_ORT_SESSION = None
_MODEL_LOCK = Lock()


class Patientdb(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    dob = models.DateField()
    state = models.CharField(max_length=50)
    gender = models.CharField(max_length=100)
    location = models.CharField(max_length=100)

    pimage = models.ImageField()

    classified = models.CharField(
        max_length=200,
        blank=True
    )

    heatmap_image = models.ImageField(
        upload_to='heatmaps/',
        blank=True,
        null=True
    )

    confidence_score = models.FloatField(
        blank=True,
        null=True
    )

    uploaded = models.DateTimeField(
        auto_now_add=True
    )

    phone_number = models.CharField(
        max_length=100
    )

    CLASS_NAMES = [
        'Benign case',
        'Malignant case',
        'Normal case'
    ]

    @staticmethod
    def get_ort_session():
        """Load ONNX runtime inference session (lazy, singleton)."""
        global _ORT_SESSION

        if _ORT_SESSION is None:
            with _MODEL_LOCK:
                if _ORT_SESSION is None:
                    import onnxruntime as ort

                    onnx_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        'model.onnx'
                    )

                    if not os.path.exists(onnx_path):
                        raise FileNotFoundError(
                            f"ONNX model not found at {onnx_path}. "
                            "Run accounts/convert_model.py during the build step."
                        )

                    logger.info("Loading ONNX model from %s", onnx_path)
                    
                    # Restrict ONNX to a single thread to minimize memory fragmentation
                    opts = ort.SessionOptions()
                    opts.intra_op_num_threads = 1
                    opts.inter_op_num_threads = 1
                    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                    
                    _ORT_SESSION = ort.InferenceSession(
                        onnx_path,
                        sess_options=opts,
                        providers=['CPUExecutionProvider']
                    )
                    logger.info("ONNX model loaded successfully.")

        return _ORT_SESSION

    def classify_image(self):
        """Run inference using ONNX Runtime (no TensorFlow at runtime)."""
        session = self.get_ort_session()

        self.pimage.open()
        img = Image.open(self.pimage).convert("RGB")
        img = img.resize((224, 224))

        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array_batch = np.expand_dims(img_array, axis=0)

        input_name = session.get_inputs()[0].name
        pred = session.run(None, {input_name: img_array_batch})[0]

        predicted_class = int(np.argmax(pred))
        confidence = float(np.max(pred) * 100)
        label = self.CLASS_NAMES[predicted_class]

        logger.info("Prediction: %s (%.2f%%)", label, confidence)

        # Return img_array (HWC, no batch) and PIL img for heatmap use
        return label, confidence, predicted_class, img_array, img

    def generate_heatmap(self, class_idx, img_array, original_img):
        """
        Coarse occlusion-sensitivity heatmap (4x4 grid = 16 + 1 passes).
        Uses 56x56 patches with stride 56 so only 17 forward passes run,
        each at batch_size=1 to keep activation memory under 250MB per pass.
        The coarse heatmap is bicubically upscaled to 224x224 for display.
        """
        session = self.get_ort_session()
        input_name = session.get_inputs()[0].name

        # Baseline confidence for the predicted class (batch=1)
        baseline_inp = np.expand_dims(img_array, axis=0)
        baseline_conf = float(
            session.run(None, {input_name: baseline_inp})[0][0][class_idx]
        )
        del baseline_inp  # free immediately

        patch_size = 56
        stride = 56
        h, w = 224, 224
        sensitivity = np.zeros((h, w), dtype=np.float32)
        counts = np.zeros((h, w), dtype=np.float32)

        # 4x4 grid = 16 positions, one forward pass at a time (batch_size=1)
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                occluded = img_array.copy()
                occluded[y:y + patch_size, x:x + patch_size, :] = 0.5
                inp = np.expand_dims(occluded, axis=0)
                conf = float(
                    session.run(None, {input_name: inp})[0][0][class_idx]
                )
                del inp, occluded  # free activation memory immediately
                drop = baseline_conf - conf
                sensitivity[y:y + patch_size, x:x + patch_size] += drop
                counts[y:y + patch_size, x:x + patch_size] += 1

        counts = np.maximum(counts, 1)
        heatmap = sensitivity / counts

        # Normalize and threshold low-activation regions
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 1e-8:
            heatmap = heatmap / np.max(heatmap)
        heatmap = np.where(heatmap > 0.3, heatmap, 0)

        # Overlay on original image using JET colormap
        original_bgr = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(original_bgr, 0.7, heatmap_colored, 0.5, 0)
        final_img = Image.fromarray(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))

        buffer = BytesIO()
        final_img.save(buffer, format="PNG")

        filename = f"heatmap_{self.pk}.png"
        self.heatmap_image.save(
            filename,
            ContentFile(buffer.getvalue()),
            save=False
        )

        logger.info("Heatmap saved for patient %s", self.pk)

    def save(self, *args, **kwargs):

        super().save(*args, **kwargs)

        # Skip if no image or already classified
        if not self.pimage or self.classified:
            return

        try:
            label, confidence, class_idx, img_array, pil_img = self.classify_image()

            # Generate heatmap (non-fatal if it fails)
            try:
                self.generate_heatmap(class_idx, img_array, pil_img)
            except Exception:
                logger.exception("Heatmap generation failed for patient %s", self.pk)

            Patientdb.objects.filter(pk=self.pk).update(
                classified=label,
                confidence_score=confidence,
                heatmap_image=self.heatmap_image.name if self.heatmap_image else None,
            )

            logger.info(
                "Classification completed for patient %s: %s (%.2f%%)",
                self.pk, label, confidence
            )

        except Exception:
            logger.exception("Classification failed for patient %s", self.pk)

            Patientdb.objects.filter(pk=self.pk).update(
                classified="Classification failed"
            )
