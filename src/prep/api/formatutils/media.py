from pathlib import Path
from typing import Any


def guess_ext(raw: bytes, src_path: str | None) -> str:
    from filetype import guess_extension

    ext = guess_extension(raw)
    if ext is None and src_path is not None:
        ext = Path(src_path).suffix.lower().lstrip(".")
    if ext is None:
        raise ValueError("Cannot guess file extension from raw bytes or source path")

    return ext


def write_to_file(entry: Any, dest: Path) -> Path:
    import shutil

    from PIL import Image as PILImage

    if isinstance(entry, PILImage.Image):
        ext = (entry.format or "PNG").lower()
        if ext == "jpeg":
            ext = "jpg"
        img = entry
        if ext == "jpg" and img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        out = dest.with_suffix(f".{ext}")
        img.save(out)
        return out

    if isinstance(entry, dict):
        raw, src_path = entry.get("bytes"), entry.get("path")
        if raw:
            out = dest.with_suffix(f".{guess_ext(raw, src_path)}")
            out.write_bytes(raw)
            return out
        if src_path and Path(src_path).exists():
            src = Path(src_path)
            out = dest.with_suffix(src.suffix.lower())
            shutil.copyfile(src, out)
            return out

    if isinstance(entry, (str, Path)) and Path(entry).exists():
        src = Path(entry)
        out = dest.with_suffix(src.suffix.lower())
        shutil.copyfile(src, out)
        return out

    raise NotImplementedError(f"Cannot write entry of type {type(entry)} to file")
