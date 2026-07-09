import os
import sys
import subprocess
import logging

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class AccountsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'accounts'

    def ready(self):
        """
        On startup, ensure model.onnx exists.
        If it doesn't, run convert_model.py in a subprocess so TensorFlow
        is isolated from the main process (avoids TF's 400 MB RAM footprint
        in the Django process itself).
        """
        onnx_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'model.onnx'
        )
        convert_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'convert_model.py'
        )

        if os.path.exists(onnx_path):
            logger.info("ONNX model already exists at %s", onnx_path)
            return

        logger.warning("model.onnx not found — running conversion subprocess...")

        # Install conversion deps quietly then convert
        try:
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install',
                 'tensorflow-cpu', 'tf2onnx', '-q', '--no-warn-script-location'],
                timeout=600
            )
            subprocess.check_call(
                [sys.executable, convert_script],
                timeout=600
            )
        except subprocess.CalledProcessError as e:
            logger.error("ONNX conversion subprocess failed: %s", e)
            # Don't crash the server — classification will fail gracefully per request
            return
        except subprocess.TimeoutExpired:
            logger.error("ONNX conversion timed out after 10 minutes")
            return

        if os.path.exists(onnx_path):
            size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
            logger.info("ONNX model ready: %.1f MB", size_mb)
        else:
            logger.error("Conversion finished but model.onnx still not found!")
