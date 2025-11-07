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
    raise ValueError(f"Unsupported processor: {name}")


def _determine_output_dir(
    input_path: Path, output_dir: Optional[str | Path], config: Config
) -> Path:
    """Determine the output directory for processed files.

    Priority: explicit output_dir > preprocess.output_dir > data.processed_dir
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

    # If input is a directory, write outputs into a subfolder named after it
    if input_path.is_dir():
        out_root = (base_out_root / input_path.name).resolve()
    else:
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
        # Get file list (only original files, processors will get images themselves if needed)
        files = _get_file_list(p, include_pdf_images=False)
    else:
        # Single file - convert PDF to images if needed (processors can use them)
        if p.suffix.lower() == ".pdf":
            from docs2synth.utils.pdf_images import convert_pdf_to_images

            pdf_dpi = cfg.get("preprocess.pdf_dpi", 300)
            convert_pdf_to_images(p, dpi=pdf_dpi, force=False)
        files = [p]

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
