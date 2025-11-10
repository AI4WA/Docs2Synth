"""Preprocess orchestration utilities.

This module centralizes the logic to run a chosen processor over a single file
or a directory of files and write schema-compliant outputs to disk.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm

from docs2synth.utils.config import Config
from docs2synth.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from PIL import Image

    _PIL_AVAILABLE = True
except (
    ImportError
):  # pragma: no cover - Pillow should be installed, but guard just in case
    Image = None  # type: ignore
    _PIL_AVAILABLE = False

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def _get_processor(name: str):
    name_lower = name.lower()
    if name_lower == "paddleocr":
        from .paddleocr import PaddleOCRProcessor

        return PaddleOCRProcessor()
    elif name_lower == "pdfplumber":
        from .pdfplumber_proc import PDFPlumberProcessor

        return PDFPlumberProcessor()
    elif name_lower == "easyocr":
        from .easyocr_proc import EasyOCRProcessor

        return EasyOCRProcessor()
    elif name_lower == "docling":
        from .docling_proc import DoclingProcessor

        return DoclingProcessor()
    raise ValueError(f"Unsupported processor: {name}")


def _determine_output_dir(
    input_path: Path, output_dir: Optional[str | Path], config: Config
) -> Path:
    """Determine the output directory for processed files.

    Priority: explicit output_dir > preprocess.output_dir > data.processed_dir

    If input is a directory, creates a subdirectory with the same name in the output directory.
    """
    if output_dir is not None:
        base_out_root = Path(output_dir).resolve()
    else:
        preprocess_output_dir = config.get("preprocess.output_dir")
        if preprocess_output_dir is not None:
            base_out_root = Path(preprocess_output_dir).resolve()
        else:
            base_out_root = Path(
                config.get("data.processed_dir", "./data/processed")
            ).resolve()

    out_root = base_out_root

    out_root.mkdir(parents=True, exist_ok=True)
    return out_root


def _get_file_list(path: Path, include_pdf_images: bool = False) -> List[Path]:
    """Get list of files to process from a path (file or directory).

    Parameters
    ----------
    path : Path
        File or directory path.
    include_pdf_images : bool
        If True, include PDF page images (1.png, 2.png, etc.) from PDF images directories.
        If False, skip PDF images directories (default: False).

    Returns
    -------
    List[Path]
        List of files to process.
    """
    if path.is_dir():
        from docs2synth.utils.pdf_images import get_pdf_images, is_pdf_images_dir

        files = []
        for fp in path.iterdir():
            # Handle PDF images directories
            if fp.is_dir():
                if is_pdf_images_dir(fp):
                    if include_pdf_images:
                        # Include all page images from this directory
                        pdf_file = fp.parent / f"{fp.name}.pdf"
                        pdf_images = get_pdf_images(pdf_file)
                        if pdf_images:
                            files.extend(pdf_images)
                            logger.debug(f"Included {len(pdf_images)} images from {fp}")
                    else:
                        logger.debug(f"Skipping PDF images directory: {fp}")
                continue
            # Only include files (not PDFs if we're including their images)
            if fp.is_file():
                # Skip PDF files if we're including their converted images
                if include_pdf_images and fp.suffix.lower() == ".pdf":
                    logger.debug(f"Skipping PDF {fp.name} (using converted images)")
                    continue
                files.append(fp)
        return sorted(files)
    return [path]


def _get_resample_filter():
    """Return the appropriate Pillow resampling filter."""
    if not _PIL_AVAILABLE:
        raise RuntimeError(
            "Pillow is required for image resizing, but it is not installed."
        )

    # Pillow>=9 exposes Image.Resampling, older versions use module-level constants
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS  # type: ignore[attr-defined]
    return Image.LANCZOS  # type: ignore[attr-defined]


def _downscale_image_if_needed(image_path: Path, max_long_edge: int) -> bool:
    """Downscale a single image in-place if its longest edge exceeds max_long_edge."""
    if not _PIL_AVAILABLE:
        raise RuntimeError(
            "Image resizing requested but Pillow is not available. "
            "Install Pillow or disable preprocess.image_resize.enabled."
        )

    try:
        with Image.open(image_path) as img:
            width, height = img.size
            longest_edge = max(width, height)
            if longest_edge <= max_long_edge or longest_edge == 0:
                return False

            scale = max_long_edge / float(longest_edge)
            new_width = max(1, int(round(width * scale)))
            new_height = max(1, int(round(height * scale)))
            if new_width == width and new_height == height:
                return False

            resample = _get_resample_filter()
            resized = img.resize((new_width, new_height), resample=resample)

            img_format = img.format or image_path.suffix.lstrip(".").upper()
            save_kwargs = {}
            if img_format and img_format.upper() in {"JPEG", "JPG"}:
                # Preserve as much quality as possible while enabling optimization
                save_kwargs["quality"] = img.info.get("quality", 95)
                save_kwargs["optimize"] = True
            elif img_format and img_format.upper() == "PNG":
                save_kwargs["optimize"] = True

            resized.save(image_path, format=img_format, **save_kwargs)

            logger.debug(
                f"Downscaled image {image_path} from {width}x{height} to {new_width}x{new_height}"
            )
            return True
    except Exception as exc:  # pragma: no cover - depends on image contents
        logger.warning(f"Failed to downscale image {image_path}: {exc}")
    return False


def _downscale_images_in_path(path: Path, max_long_edge: int) -> None:
    """Downscale images under a path (file or directory) when needed."""
    if max_long_edge <= 0:
        logger.warning(
            "preprocess.image_resize.max_long_edge must be positive; skipping downscale"
        )
        return

    if path.is_file():
        if path.suffix.lower() in _IMAGE_EXTENSIONS:
            _downscale_image_if_needed(path, max_long_edge)
        return

    if not path.exists():
        return

    processed = 0
    modified = 0
    for file_path in path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in _IMAGE_EXTENSIONS:
            processed += 1
            if _downscale_image_if_needed(file_path, max_long_edge):
                modified += 1

    if processed:
        logger.info(
            f"Checked {processed} image(s) in {path} for resizing; downscaled {modified}"
        )


def _prewarm_processor(
    proc, processor: str, files: List[Path], lang: Optional[str], device: Optional[str]
) -> None:
    """Pre-warm processor initialization, especially for PaddleOCR."""
    if len(files) == 0 or processor.lower() != "paddleocr":
        return

    logger.info("Initializing PaddleOCR...")
    try:
        kwargs = {}
        if lang is not None:
            kwargs["lang"] = lang
        if device is not None:
            kwargs["device"] = device
        proc._init_ocr(  # type: ignore[attr-defined]
            lang_override=kwargs.get("lang"),
            device_override=kwargs.get("device"),
        )
        logger.info("PaddleOCR initialization completed")
    except Exception as e:
        logger.warning(
            f"Pre-initialization failed, will retry during file processing: {e}"
        )


def _process_single_file(
    file_path: Path,
    processor,
    processor_name: str,
    output_root: Path,
    lang: Optional[str],
    device: Optional[str],
) -> Optional[Path]:
    """Process a single file and return output path if successful."""
    kwargs = {}
    if lang is not None:
        kwargs["lang"] = lang
    if device is not None:
        kwargs["device"] = device

    result = processor.process(str(file_path), **kwargs)  # type: ignore[arg-type]

    out_filename = f"{file_path.stem}_{processor_name.lower()}.json"
    out_path = output_root / out_filename
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(result.to_json(indent=2))

    return out_path


def run_preprocess(
    path: str | Path,
    *,
    processor: str = "paddleocr",
    output_dir: Optional[str | Path] = None,
    lang: Optional[str] = None,
    device: Optional[str] = None,
    config: Optional[Config] = None,
) -> Tuple[int, int, List[Path]]:
    """Run preprocessing on a file or directory.

    Parameters
    ----------
    path : str | Path
        File or directory to process.
    processor : str
        Processor name (default: "paddleocr").
    output_dir : Optional[str | Path]
        Directory to write outputs (defaults to config.data.processed_dir).
    lang : Optional[str]
        Optional language override passed to the processor (if supported).
    config : Optional[Config]
        Configuration instance; if None, a default will be used.

    Returns
    -------
    (num_success, num_failed, output_paths)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")

    cfg = config or Config()
    out_root = _determine_output_dir(p, output_dir, cfg)

    # If input is a directory, convert PDFs to images first (for processors that need them)
    if p.is_dir():
        from docs2synth.utils.pdf_images import convert_pdfs_in_directory

        pdf_dpi = cfg.get("preprocess.pdf_dpi", 300)
        logger.info(f"Pre-converting PDFs to images (DPI: {pdf_dpi})...")
        convert_pdfs_in_directory(p, dpi=pdf_dpi, force=False)

        resize_cfg = cfg.get("preprocess.image_resize", {})
        resize_enabled = isinstance(resize_cfg, dict) and resize_cfg.get(
            "enabled", False
        )
        if resize_enabled:
            if not _PIL_AVAILABLE:
                raise RuntimeError(
                    "preprocess.image_resize.enabled is True, but Pillow is not installed."
                )
            max_long_edge = int(resize_cfg.get("max_long_edge", 1280))
            _downscale_images_in_path(p, max_long_edge)

        # Get file list (only original files, processors will get images themselves if needed)
        files = _get_file_list(p, include_pdf_images=False)
    else:
        # Single file - convert PDF to images if needed (processors can use them)
        if p.suffix.lower() == ".pdf":
            from docs2synth.utils.pdf_images import (
                convert_pdf_to_images,
                get_pdf_images_dir,
            )

            pdf_dpi = cfg.get("preprocess.pdf_dpi", 300)
            convert_pdf_to_images(p, dpi=pdf_dpi, force=False)
            resize_cfg = cfg.get("preprocess.image_resize", {})
            resize_enabled = isinstance(resize_cfg, dict) and resize_cfg.get(
                "enabled", False
            )
            if resize_enabled:
                if not _PIL_AVAILABLE:
                    raise RuntimeError(
                        "preprocess.image_resize.enabled is True, but Pillow is not installed."
                    )
                max_long_edge = int(resize_cfg.get("max_long_edge", 1280))
                images_dir = get_pdf_images_dir(p)
                _downscale_images_in_path(images_dir, max_long_edge)
        files = [p]
        resize_cfg = cfg.get("preprocess.image_resize", {})
        resize_enabled = isinstance(resize_cfg, dict) and resize_cfg.get(
            "enabled", False
        )
        if resize_enabled and p.suffix.lower() in _IMAGE_EXTENSIONS:
            if not _PIL_AVAILABLE:
                raise RuntimeError(
                    "preprocess.image_resize.enabled is True, but Pillow is not installed."
                )
            max_long_edge = int(resize_cfg.get("max_long_edge", 1280))
            _downscale_images_in_path(p, max_long_edge)

    proc = _get_processor(processor)

    _prewarm_processor(proc, processor, files, lang, device)

    num_success = 0
    num_failed = 0
    outputs: List[Path] = []

    # Use tqdm to show progress bar, write to stderr to avoid conflicts with stdout
    progress_bar = tqdm(
        files,
        desc=f"Processing ({processor})",
        unit="file",
        disable=len(files) == 1,  # Disable for single file to keep output clean
        file=sys.stderr,  # Use stderr to avoid conflicts with processor output
        dynamic_ncols=True,  # Adjust width based on terminal
    )

    for f in progress_bar:
        progress_bar.set_postfix(
            file=f.name[:30] + "..." if len(f.name) > 30 else f.name
        )
        progress_bar.refresh()  # Force refresh before processing
        try:
            out_path = _process_single_file(f, proc, processor, out_root, lang, device)
            num_success += 1
            outputs.append(out_path)
            # Use tqdm.write to avoid interfering with progress bar
            tqdm.write(f"Processed {f.name} -> {out_path.name}", file=sys.stderr)
        except Exception as e:  # pragma: no cover - exercised via CLI typically
            num_failed += 1
            # Use tqdm.write for error messages too
            tqdm.write(f"Failed processing {f.name}: {e}", file=sys.stderr)
            logger.exception(f"Failed processing {f}: {e}")

    return num_success, num_failed, outputs
