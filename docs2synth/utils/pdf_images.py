"""Utilities for managing PDF-to-image conversions.

This module provides functions to save PDF pages as images in a structured way
and retrieve them when needed (e.g., for QA generation).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from docs2synth.utils.logging import get_logger

logger = get_logger(__name__)


def get_pdf_images_dir(pdf_path: str | Path) -> Path:
    """Get the directory where PDF page images should be stored.

    Structure: {pdf_dir}/{pdf_stem}/
    Example: /path/to/document.pdf -> /path/to/document/

    Parameters
    ----------
    pdf_path : str | Path
        Path to the PDF file.

    Returns
    -------
    Path
        Directory path for storing page images.
    """
    pdf_path = Path(pdf_path)
    return pdf_path.parent / pdf_path.stem


def save_pdf_images(image_paths: List[str], pdf_path: str | Path) -> List[Path]:
    """Save PDF page images to a structured directory.

    Images are saved as: {pdf_dir}/{pdf_stem}/1.png, 2.png, ...

    Parameters
    ----------
    image_paths : List[str]
        List of temporary image file paths (from PDF conversion).
    pdf_path : str | Path
        Path to the original PDF file.

    Returns
    -------
    List[Path]
        List of saved image file paths.
    """
    pdf_path = Path(pdf_path)
    images_dir = get_pdf_images_dir(pdf_path)
    images_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    for idx, temp_image_path in enumerate(image_paths, start=1):
        saved_path = images_dir / f"{idx}.png"
        # Copy the temporary image to the final location
        import shutil

        shutil.copy2(temp_image_path, saved_path)
        saved_paths.append(saved_path)

    return saved_paths


def get_pdf_images(pdf_path: str | Path) -> Optional[List[Path]]:
    """Get list of page images for a PDF file if they exist.

    Parameters
    ----------
    pdf_path : str | Path
        Path to the PDF file.

    Returns
    -------
    Optional[List[Path]]
        List of image paths if they exist, None otherwise.
        Images are sorted by page number (1.png, 2.png, ...).
    """
    pdf_path = Path(pdf_path)
    images_dir = get_pdf_images_dir(pdf_path)

    if not images_dir.exists():
        return None

    # Find all numbered PNG files (1.png, 2.png, etc.)
    image_files = []
    for img_file in sorted(images_dir.glob("*.png")):
        # Check if filename is a number
        try:
            page_num = int(img_file.stem)
            image_files.append((page_num, img_file))
        except ValueError:
            # Skip non-numbered files
            continue

    if not image_files:
        return None

    # Sort by page number and return paths
    image_files.sort(key=lambda x: x[0])
    return [path for _, path in image_files]


def is_pdf_images_dir(path: Path) -> bool:
    """Check if a path is a PDF images directory.

    A PDF images directory is a directory that:
    1. Has the same name as a PDF file in its parent directory
    2. Contains numbered PNG files (1.png, 2.png, ...)

    Parameters
    ----------
    path : Path
        Path to check.

    Returns
    -------
    bool
        True if this appears to be a PDF images directory.
    """
    if not path.is_dir():
        return False

    # Check if parent directory contains a PDF with matching stem
    parent = path.parent
    pdf_file = parent / f"{path.name}.pdf"
    if not pdf_file.exists():
        return False

    # Check if directory contains numbered PNG files
    png_files = list(path.glob("*.png"))
    if not png_files:
        return False

    # Check if at least one file has a numeric name
    for png_file in png_files:
        try:
            int(png_file.stem)
            return True
        except ValueError:
            continue

    return False


def convert_pdf_to_images(
    pdf_path: str | Path, dpi: int = 300, force: bool = False
) -> Optional[List[Path]]:
    """Convert a PDF file to page images.

    Images are saved to {pdf_dir}/{pdf_stem}/1.png, 2.png, ...
    If images already exist and force=False, returns existing images.

    Parameters
    ----------
    pdf_path : str | Path
        Path to the PDF file.
    dpi : int
        DPI (resolution) for conversion (default: 300).
    force : bool
        If True, reconvert even if images already exist (default: False).

    Returns
    -------
    Optional[List[Path]]
        List of image file paths, or None if conversion failed.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists() or not pdf_path.suffix.lower() == ".pdf":
        return None

    images_dir = get_pdf_images_dir(pdf_path)
    existing_images = get_pdf_images(pdf_path)

    # Return existing images if they exist and not forcing
    if existing_images and not force:
        logger.debug(f"Using existing images for {pdf_path}")
        return existing_images

    # Check if PyMuPDF is available
    try:
        import fitz  # PyMuPDF  # type: ignore
    except ImportError:
        logger.error("PyMuPDF is not installed. Cannot convert PDF to images.")
        return None

    try:
        doc = fitz.open(str(pdf_path))  # type: ignore
        images_dir.mkdir(parents=True, exist_ok=True)
        image_paths: List[Path] = []

        try:
            logger.info(f"Converting PDF {pdf_path.name} to images (DPI: {dpi})...")
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # Convert page to image with specified DPI
                mat = fitz.Matrix(dpi / 72, dpi / 72)  # type: ignore
                pix = page.get_pixmap(matrix=mat)
                # Save as numbered PNG file
                image_filename = f"{page_num + 1}.png"
                image_path = images_dir / image_filename
                pix.save(str(image_path))
                image_paths.append(image_path)
                logger.debug(f"Converted page {page_num + 1}/{len(doc)}")

            logger.info(f"Converted {len(image_paths)} pages to {images_dir}")
            return image_paths
        finally:
            doc.close()
    except Exception as e:
        logger.error(f"Failed to convert PDF {pdf_path} to images: {e}")
        return None


def convert_pdfs_in_directory(
    directory: str | Path, dpi: int = 300, force: bool = False
) -> int:
    """Convert all PDF files in a directory to images.

    Parameters
    ----------
    directory : str | Path
        Directory containing PDF files.
    dpi : int
        DPI (resolution) for conversion (default: 300).
    force : bool
        If True, reconvert even if images already exist (default: False).

    Returns
    -------
    int
        Number of PDFs successfully converted.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return 0

    pdf_files = [
        f for f in directory.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"
    ]
    if not pdf_files:
        return 0

    logger.info(f"Found {len(pdf_files)} PDF file(s) in {directory}")

    converted = 0
    for pdf_file in sorted(pdf_files):
        result = convert_pdf_to_images(pdf_file, dpi=dpi, force=force)
        if result:
            converted += 1

    logger.info(f"Converted {converted}/{len(pdf_files)} PDF file(s)")
    return converted
