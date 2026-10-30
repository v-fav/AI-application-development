"""Перевірка та підготовка зображень для мультимодальної моделі."""
from dataclasses import dataclass
from io import BytesIO
import os
import warnings
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = int(float(os.getenv("UPLOAD_MAX_MB", "10")) * 1024 * 1024)
MAX_SIDE = int(os.getenv("IMAGE_MAX_SIDE", "1600"))
MAX_PIXELS = int(os.getenv("IMAGE_MAX_PIXELS", "30000000"))
MIN_SIDE = int(os.getenv("IMAGE_MIN_SIDE", "500"))

class ImageError(ValueError):
    pass

@dataclass
class PreparedImage:
    data: bytes
    mime: str
    original: dict
    sent: dict

def prepare(content: bytes) -> PreparedImage:
    if not content:
        raise ImageError("Файл порожній.")
    if len(content) > MAX_BYTES:
        raise ImageError(f"Файл завеликий: {len(content)/1024/1024:.1f} МБ. Ліміт — {MAX_BYTES/1024/1024:.1f} МБ.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(BytesIO(content))
            image.verify()
        image = Image.open(BytesIO(content))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ImageError("Файл не є коректним зображенням або має небезпечну роздільність.") from exc

    width, height = image.size
    if width * height > MAX_PIXELS:
        raise ImageError(f"Зображення має забагато пікселів: {width}×{height}.")
    if min(width, height) < MIN_SIDE:
        raise ImageError(f"Знімок замалий ({width}×{height}). Будь ласка, перезніміть документ чіткіше.")

    original = {"width": width, "height": height, "bytes": len(content)}
    if max(width, height) > MAX_SIDE:
        scale = MAX_SIDE / max(width, height)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)

    # PNG зберігає дрібний текст без артефактів JPEG.
    out = BytesIO()
    image.save(out, format="PNG", optimize=True)
    data = out.getvalue()
    return PreparedImage(
        data=data,
        mime="image/png",
        original=original,
        sent={"width": image.width, "height": image.height, "bytes": len(data)},
    )
