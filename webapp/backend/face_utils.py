"""
Lightweight face-presence check used only by the game (NOT by the model).

Before we spend a prediction on a webcam frame, we do a quick, cheap check
with OpenCV's built-in Haar cascade face detector so the game can tell the
player things like "we can't see your face" or "please make sure only one
face is visible" without ever consuming a heart for those cases.

This is deliberately separate from the emotion model itself.
"""

import cv2
import numpy as np

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def _iou(a, b):
    """Intersection-over-union of two (x, y, w, h) rectangles."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _merge_overlapping(rects, iou_threshold=0.3):
    """Collapse detections that clearly belong to the same face.

    Haar cascades can fire more than once for a single real face (slightly
    offset boxes at neighboring scales) even after ``minNeighbors``
    filtering. Without this step those get miscounted as separate people.
    """
    if len(rects) <= 1:
        return list(rects)

    rects = sorted(rects, key=lambda r: r[2] * r[3], reverse=True)
    kept = []
    for rect in rects:
        if all(_iou(rect, k) < iou_threshold for k in kept):
            kept.append(rect)
    return kept


def _looks_like_skin(bgr_image, rect, min_skin_ratio=0.30):
    """Cheap first-pass sanity check that a detected box plausibly contains
    a face: rejects boxes with too little skin-tone-like color content
    (YCrCb space, fairly lighting-robust). Not sufficient on its own — see
    _has_eye_pattern() below — since skin-toned clothing/fabric can pass a
    color-only check, but it's a fast way to reject obviously-wrong regions
    (e.g. gray/white ceiling tiles) before paying for the eye-cascade pass.
    """
    x, y, w, h = rect
    crop = bgr_image[y : y + h, x : x + w]
    if crop.size == 0:
        return False

    ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
    _, cr, cb = cv2.split(ycrcb)
    skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    return float(skin_mask.mean()) >= min_skin_ratio


def _has_eye_pattern(gray_image, rect):
    """Verify at least one eye-like pattern exists inside a candidate box.

    This is the stronger of the two sanity checks: color alone isn't
    enough to rule out false positives, since skin-toned/tan fabric (e.g.
    a beige plaid shirt) can pass a color-only check just as easily as real
    skin. An eye cascade is far more specific — plain fabric or ceiling
    tiles essentially never produce an eye-shaped detection, whereas any
    real face (even at an angle) usually shows at least one.
    """
    x, y, w, h = rect
    crop = gray_image[y : y + h, x : x + w]
    if crop.size == 0:
        return False

    equalized = cv2.equalizeHist(crop)
    min_eye_size = max(10, int(min(w, h) * 0.12))
    eyes = _eye_cascade.detectMultiScale(
        equalized, scaleFactor=1.05, minNeighbors=3, minSize=(min_eye_size, min_eye_size)
    )
    return len(eyes) >= 1


def _looks_like_face(bgr_image, gray_image, rect):
    """Combined sanity check used to reject Haar cascade false positives
    before a detected box is trusted as a real face."""
    return _looks_like_skin(bgr_image, rect) and _has_eye_pattern(gray_image, rect)


def _raw_detect(gray, scale_factor, min_neighbors, min_size_frac):
    shortest_side = min(gray.shape[:2])
    min_size = max(40, int(shortest_side * min_size_frac))
    equalized = cv2.equalizeHist(gray)
    faces = _face_cascade.detectMultiScale(
        equalized, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(min_size, min_size)
    )
    return list(faces)


def _detect_full_resolution(bgr_image, gray):
    """Face detection tuned for real webcam-resolution frames.

    Runs a balanced first pass, and only falls back to a much more
    sensitive (but noisier) second pass if the first pass finds nothing —
    that way normal, well-lit captures never pay the extra false-positive
    risk of the sensitive settings, but a player who was wrongly told
    "no face detected" gets a second, more forgiving look before the game
    gives up on the frame.
    """
    faces = _merge_overlapping(_raw_detect(gray, scale_factor=1.1, min_neighbors=6, min_size_frac=0.12))
    if len(faces) == 0:
        faces = _merge_overlapping(_raw_detect(gray, scale_factor=1.05, min_neighbors=3, min_size_frac=0.08))
        # The sensitive pass is noisy enough to regularly fire on non-faces
        # entirely (ceiling tiles, plaid/checkered clothing, etc. — patterns
        # that share Haar-like edge structure with a face). Verify each
        # candidate actually looks like a face before trusting it.
        faces = [f for f in faces if _looks_like_face(bgr_image, gray, f)]
        # The sensitive fallback pass is noisy enough that a spurious second
        # "face" is far more likely than a real second person suddenly
        # appearing — only the balanced first pass above is trusted to
        # report genuine multiple faces.
        if len(faces) > 1:
            faces = [max(faces, key=lambda r: r[2] * r[3])]
    return faces


def _detect_small_image(bgr_image, gray):
    """Face detection tuned for small, low-res images (e.g. 48x48 dataset
    thumbnails), which the cascade otherwise fails to find anything in."""
    shortest_side = min(gray.shape[:2])
    scale = 200 / shortest_side
    upscaled_gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    upscaled_bgr = cv2.resize(bgr_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    faces = _merge_overlapping(_raw_detect(upscaled_gray, scale_factor=1.08, min_neighbors=4, min_size_frac=0.15))
    # This path already uses a fairly permissive min_neighbors, so it can
    # false-positive the same way the full-resolution fallback pass does —
    # apply the same face sanity check before trusting a detection.
    faces = [f for f in faces if _looks_like_face(upscaled_bgr, upscaled_gray, f)]

    # Detection ran on the upscaled image, so box coordinates are in that
    # scaled-up space. Callers (extract_largest_face) crop from the
    # *original*, non-upscaled image, so the boxes must be scaled back down
    # to original-image coordinates here before returning — otherwise the
    # crop comes out wrong/misaligned for every small image.
    faces = [tuple(int(round(v / scale)) for v in f) for f in faces]
    return faces


def _detect_faces(image_bytes: bytes):
    """Decode image bytes and return (bgr_image, [(x, y, w, h), ...]).

    Shared by count_faces() and extract_largest_face() so both work off the
    exact same detection pass instead of duplicating the decode/scale logic.
    ``bgr_image`` is None if the bytes couldn't be decoded as an image.
    """
    file_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return None, []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shortest_side = min(gray.shape[:2])

    if shortest_side < 200:
        faces = _detect_small_image(img, gray)
    else:
        faces = _detect_full_resolution(img, gray)

    return img, faces


def count_faces(image_bytes: bytes) -> int:
    """Return the number of distinct faces detected in the given image bytes."""
    _, faces = _detect_faces(image_bytes)
    return len(faces)


def extract_largest_face(image_bytes: bytes, margin: float = 0.05, grayscale: bool = True):
    """Crop out the (largest) detected face and return it as JPEG bytes.

    The model should see just the face, not the whole webcam frame (desk,
    walls, other people in the background, etc.), so this is what gets
    handed to the prediction service instead of the raw capture.

    ``margin`` pads the tight Haar cascade box by this fraction of the
    face's width/height on every side. This is kept SMALL (not the ~25%
    you might expect) so the crop matches the training data: the model was
    trained on FER2013, whose images are already tight, hair/background-free
    crops (eyebrows-to-chin, basically). A generous margin here would feed
    the model hair/collar/background it never saw during training, which
    hurts confidence even on an otherwise-correct prediction.

    ``grayscale`` converts the crop to grayscale (then replicates it back to
    3 channels) for the same reason — FER2013 images are single-channel
    grayscale, so matching that at inference removes another train/inference
    mismatch.

    Returns the cropped face encoded as JPEG bytes, or None if no face /
    the image couldn't be decoded.
    """
    img, faces = _detect_faces(image_bytes)
    if img is None or len(faces) == 0:
        return None

    # Largest-area face wins (mirrors the "trust the biggest box" behaviour
    # already used for the sensitive fallback pass above).
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])

    img_h, img_w = img.shape[:2]
    pad_x, pad_y = int(w * margin), int(h * margin)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)

    face_crop = img[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None

    if grayscale:
        gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        # Convert back to a 3-channel image (all channels identical) rather
        # than staying single-channel: the model's first conv layer expects
        # 3 input channels (it's a pretrained ImageNet backbone), and this
        # is exactly what happens to FER2013's grayscale images too, since
        # predict_image() calls Image.open(...).convert("RGB") on them.
        face_crop = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2BGR)

    success, encoded = cv2.imencode(".jpg", face_crop)
    if not success:
        return None

    return encoded.tobytes()
