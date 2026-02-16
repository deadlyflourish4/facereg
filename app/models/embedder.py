"""
Face Embedding Module
Uses ArcFace (ONNX Runtime) to extract 512D face embeddings.
Self-downloads the model on first use.
"""
import os
import logging
import numpy as np
import cv2
import onnxruntime

logger = logging.getLogger(__name__)

# ArcFace model config
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "models")
ARCFACE_FILE = os.path.join(MODELS_DIR, "arcface_r100.onnx")
ARCFACE_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"

_session = None
_input_name = None
_output_names = None
_input_size = (112, 112)


def _download_arcface():
    """Download ArcFace model from insightface releases."""
    import zipfile
    import tempfile
    import urllib.request

    os.makedirs(MODELS_DIR, exist_ok=True)

    logger.info("Downloading ArcFace model... (first time only, ~300MB)")

    zip_path = os.path.join(tempfile.gettempdir(), "buffalo_l.zip")

    # Download with progress
    def _progress(block_num, block_size, total_size):
        pct = min(100, block_num * block_size * 100 // max(total_size, 1))
        if block_num % 200 == 0:
            logger.info(f"  Download progress: {pct}%")

    urllib.request.urlretrieve(ARCFACE_URL, zip_path, _progress)
    logger.info("Download complete, extracting ArcFace model...")

    # Extract only the recognition model (w600k_r50.onnx)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            # Look for recognition model (112x112 input = ArcFace)
            if name.endswith(".onnx"):
                data = zf.read(name)
                # Test if it's the recognition model by checking ONNX input shape
                import tempfile as tf
                tmp = os.path.join(tf.gettempdir(), os.path.basename(name))
                with open(tmp, "wb") as f:
                    f.write(data)
                try:
                    sess = onnxruntime.InferenceSession(tmp, None)
                    inp = sess.get_inputs()[0]
                    shape = inp.shape
                    if len(shape) == 4 and shape[2] == 112 and shape[3] == 112:
                        # This is the recognition model!
                        with open(ARCFACE_FILE, "wb") as f:
                            f.write(data)
                        logger.info(f"  Extracted ArcFace model: {name} → {ARCFACE_FILE}")
                        os.remove(tmp)
                        break
                except Exception:
                    pass
                finally:
                    if os.path.exists(tmp):
                        os.remove(tmp)

    # Cleanup
    if os.path.exists(zip_path):
        os.remove(zip_path)

    if not os.path.exists(ARCFACE_FILE):
        raise RuntimeError("Failed to extract ArcFace model from zip")

    logger.info("ArcFace model ready!")


def _get_session():
    """Lazy-load ONNX session for ArcFace."""
    global _session, _input_name, _output_names

    if _session is not None:
        return _session

    if not os.path.exists(ARCFACE_FILE):
        _download_arcface()

    logger.info(f"Loading ArcFace model from {ARCFACE_FILE}")
    _session = onnxruntime.InferenceSession(
        ARCFACE_FILE,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    inp = _session.get_inputs()[0]
    _input_name = inp.name
    _output_names = [o.name for o in _session.get_outputs()]
    logger.info(f"ArcFace loaded: input={inp.shape}, outputs={_output_names}")

    return _session


def get_embedding(aligned_face):
    """Extract 512D embedding from an aligned face (112x112 RGB numpy array).

    Args:
        aligned_face: numpy array (112, 112, 3) RGB

    Returns:
        numpy array of shape (512,), L2-normalized. None if failed.
    """
    try:
        session = _get_session()

        # Ensure correct size
        if aligned_face.shape[:2] != _input_size:
            aligned_face = cv2.resize(aligned_face, _input_size)

        # ArcFace expects BGR input
        face_bgr = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)

        # Preprocess: blob (NCHW, float32, normalized)
        blob = cv2.dnn.blobFromImage(
            face_bgr, 1.0 / 127.5, _input_size, (127.5, 127.5, 127.5), swapRB=True
        )

        # Run inference
        outputs = session.run(_output_names, {_input_name: blob})
        embedding = outputs[0].flatten()

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return None


def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between two normalized vectors."""
    return float(np.dot(vec_a, vec_b))
