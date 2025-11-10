"""Batch QA verification for preprocessed documents with QA pairs.

This module processes JSON files containing QA pairs to verify their quality.
It runs configured verifiers (meaningful, correctness) on each QA pair and
adds verification results to the JSON files.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image as PILImage
from tqdm import tqdm

from docs2synth.preprocess.schema import DocumentProcessResult, QAPair, RunMetadata
from docs2synth.qa.config import QAVerificationConfig
from docs2synth.qa.verifiers import create_verifier
from docs2synth.utils.pdf_images import get_pdf_images

logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf", ".tiff", ".tif", ".bmp"}


def find_image_for_json(json_path: Path, image_dirs: List[Path]) -> Optional[Path]:
    """Find the corresponding image file for a JSON file.

    Args:
        json_path: Path to the JSON file
        image_dirs: List of directories to search for images

    Returns:
        Path to the image file if found, None otherwise
    """
    # Extract base name (remove processor suffix like _paddleocr)
    json_stem = json_path.stem

    # Try to remove common processor suffixes
    for suffix in ["_paddleocr", "_easyocr", "_tesseract", "_pdfplumber", "_docling"]:
        if json_stem.endswith(suffix):
            json_stem = json_stem[: -len(suffix)]
            break

    # Search for image files with matching stem
    for image_dir in image_dirs:
        if not image_dir.exists():
            continue

        for ext in IMAGE_EXTENSIONS:
            image_path = image_dir / f"{json_stem}{ext}"
            if image_path.exists():
                return image_path

    logger.warning(f"Could not find image file for JSON {json_path.name}")
    return None


def _load_document_images(
    image_path: Path,
) -> Tuple[Optional[dict[int, PILImage.Image]], Optional[PILImage.Image], bool]:
    """Load images for a document (handles PDF multi-page and single image cases).

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (page_images dict, single image, is_pdf flag)
        Returns (None, None, False) if loading fails
    """
    is_pdf = image_path.suffix.lower() == ".pdf"
    page_images: Optional[dict[int, PILImage.Image]] = None
    image: Optional[PILImage.Image] = None

    if is_pdf:
        # For PDF files, load pre-converted page images
        pdf_images = get_pdf_images(image_path)

        if pdf_images:
            page_images = {}
            for page_idx, page_image_path in enumerate(pdf_images):
                try:
                    page_images[page_idx] = PILImage.open(page_image_path)
                    logger.debug(
                        f"Loaded PDF page {page_idx + 1}: {page_images[page_idx].size}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to load PDF page {page_idx + 1}: {e}")
            logger.info(f"Loaded {len(page_images)} PDF pages")
        else:
            logger.error(
                f"PDF page images not found for {image_path.name}. "
                f"Please run preprocessing first to convert PDF to images."
            )
            return None, None, True

    # Load single image for non-PDF files
    if not is_pdf:
        try:
            image = PILImage.open(image_path)
            logger.info(f"Loaded image: {image.size}")
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None, None, False

    return page_images, image, is_pdf


def _get_object_image(
    obj: Any,
    obj_id: int,
    is_pdf: bool,
    page_images: Optional[dict[int, PILImage.Image]],
    image: Optional[PILImage.Image],
) -> Optional[PILImage.Image]:
    """Get the appropriate image for an object based on its page field.

    Args:
        obj: DocumentObject instance
        obj_id: Object ID for logging
        is_pdf: Whether document is a PDF
        page_images: Dict of page images (for PDFs)
        image: Single image (for non-PDFs)

    Returns:
        PIL Image object or None if not found
    """
    if is_pdf and page_images is not None:
        # Use page-specific image if available
        obj_page = obj.page if obj.page is not None else 0
        obj_image = page_images.get(obj_page)
        if obj_image is None:
            logger.warning(
                f"Page {obj_page} image not found for object {obj_id}, "
                f"using first available page"
            )
            obj_image = next(iter(page_images.values())) if page_images else None
        return obj_image
    elif not is_pdf and image is not None:
        # Single image for non-PDF files
        return image
    else:
        logger.error(
            f"Could not determine image for object {obj_id} "
            f"(is_pdf={is_pdf}, page_images={page_images is not None}, image={image is not None})"
        )
        return None


def _prepare_verifier_kwargs(
    verifier_type: str,
    qa_pair: QAPair,
    context: str | None,
    image_obj: Any,
    verifier_config: Any,
) -> Tuple[dict[str, Any], bool]:
    """Prepare verification kwargs for a specific verifier type.

    Args:
        verifier_type: Type of verifier (meaningful, correctness, etc.)
        qa_pair: QA pair to verify
        context: Document context
        image_obj: Document image object
        verifier_config: Verifier configuration

    Returns:
        Tuple of (verify_kwargs dict, should_skip bool)
    """
    verify_kwargs: dict[str, Any] = {}

    # Add temperature and max_tokens from config if available
    if verifier_config.temperature is not None:
        verify_kwargs["temperature"] = verifier_config.temperature
    if verifier_config.max_tokens is not None:
        verify_kwargs["max_tokens"] = verifier_config.max_tokens

    # Add image
    verify_kwargs["image"] = image_obj

    if verifier_type == "meaningful":
        # Meaningful verifier needs context or image
        if context is None and image_obj is None:
            return {}, True
        if context is not None:
            verify_kwargs["context"] = context
    elif verifier_type == "correctness":
        # Correctness verifier needs answer
        if qa_pair.answer is None:
            return {}, True

    return verify_kwargs, False


def _has_existing_verification(
    qa_pair: QAPair,
    verifier_type: str,
    provider: str,
    model: Optional[str],
) -> bool:
    """Check if QA pair already has verification for a given verifier_type + provider + model combination.

    Args:
        qa_pair: QA pair to check
        verifier_type: Verifier type to check (e.g., "meaningful", "correctness")
        provider: Provider name to check
        model: Model name to check (None means "default")

    Returns:
        True if verification exists with matching verifier_type + provider + model, False otherwise
    """
    if not qa_pair.verification:
        return False

    # Normalize model: None or empty string means "default"
    expected_model = model or "default"

    # Check if this verifier_type already exists in verification results
    if verifier_type not in qa_pair.verification:
        return False

    existing_result = qa_pair.verification[verifier_type]
    existing_provider = existing_result.get("provider")
    existing_model = (
        existing_result.get("model") or "default"
    )  # Normalize None/empty to "default"

    # Check if provider and model match
    if existing_provider == provider and existing_model == expected_model:
        logger.debug(
            f"Found existing verification: verifier_type='{verifier_type}', "
            f"provider='{existing_provider}', model='{existing_model}' "
            f"(looking for provider='{provider}', model='{expected_model}')"
        )
        return True

    return False


def _verify_qa_pair(
    qa_pair: QAPair,
    verifiers: List[Tuple[str, Any]],
    context: str,
    image_obj: Any,
) -> Dict[str, Dict[str, Any]]:
    """Verify a single QA pair using configured verifiers.

    Args:
        qa_pair: QA pair to verify
        verifiers: List of (verifier_type, verifier_instance) tuples
        context: Document context
        image_obj: Document image object

    Returns:
        Dictionary mapping verifier_type to verification result
    """
    verification_results: Dict[str, Dict[str, Any]] = {}

    for verifier_type, verifier_instance in verifiers:
        try:
            # Get verifier config from the instance
            verifier_config = getattr(verifier_instance, "_config", None)

            if verifier_config:
                verifier_provider = verifier_config.provider
                verifier_model = verifier_config.model or "default"
            else:
                # Fallback if config is not available
                verifier_provider = "unknown"
                verifier_model = "default"

            # Check if verification already exists for this exact combination
            if _has_existing_verification(
                qa_pair,
                verifier_type,
                verifier_provider,
                verifier_config.model if verifier_config else None,
            ):
                logger.info(
                    f"Skipping {verifier_type} verifier "
                    f"(provider: '{verifier_provider}', model: '{verifier_model}'): already exists"
                )
                continue

            # Prepare kwargs
            verify_kwargs, should_skip = _prepare_verifier_kwargs(
                verifier_type, qa_pair, context, image_obj, verifier_config
            )

            if should_skip:
                logger.debug(
                    f"Skipping {verifier_type} verifier (missing required inputs)"
                )
                continue

            # Run verification
            result = verifier_instance.verify(
                question=qa_pair.question,
                answer=qa_pair.answer,
                **verify_kwargs,
            )

            # Add verifier metadata
            result["verifier_type"] = verifier_type
            if verifier_config:
                result["provider"] = verifier_config.provider
                result["model"] = verifier_config.model or "default"

            verification_results[verifier_type] = result

            logger.debug(
                f"Verified QA pair with {verifier_type}: "
                f"{result.get('response', 'Unknown')}"
            )

        except Exception as e:
            logger.error(f"Failed to verify QA pair with {verifier_type}: {e}")
            continue

    return verification_results


def _create_verifiers(
    verification_config: QAVerificationConfig,
    config_path: str | None = None,
) -> List[Tuple[str, Any]]:
    """Create verifier instances from configuration.

    Args:
        verification_config: QA verification configuration
        config_path: Path to config.yml for loading API keys

    Returns:
        List of (verifier_type, verifier_instance) tuples
    """
    verifiers = []

    for verifier_config in verification_config.verifiers:
        try:
            verifier = create_verifier(verifier_config, config_path=config_path)
            # Store config in verifier for later use
            verifier._config = verifier_config
            verifiers.append((verifier_config.verifier_type, verifier))
            logger.info(
                f"Created {verifier_config.verifier_type} verifier "
                f"({verifier_config.provider}/{verifier_config.model or 'default'})"
            )
        except Exception as e:
            logger.error(
                f"Failed to create {verifier_config.verifier_type} verifier: {e}"
            )
            continue

    return verifiers


def _collect_verifier_metadata(
    verifiers: List[Tuple[str, Any]]
) -> List[Dict[str, Any]]:
    """Collect metadata about verifiers used in the verification run.

    Args:
        verifiers: List of (verifier_type, verifier_instance) tuples

    Returns:
        List of dictionaries with verifier metadata
    """
    metadata = []
    for verifier_type, verifier_instance in verifiers:
        verifier_config = getattr(verifier_instance, "_config", None)
        if verifier_config:
            metadata.append(
                {
                    "verifier_type": verifier_type,
                    "provider": verifier_config.provider,
                    "model": verifier_config.model or "default",
                }
            )
    return metadata


def process_document_verification(
    json_path: Path,
    image_path: Path,
    verification_config: QAVerificationConfig,
    config_path: str | None = None,
) -> Tuple[int, int, int]:
    """Process a single document to verify QA pairs.

    Args:
        json_path: Path to the JSON file with QA pairs
        image_path: Path to the corresponding image file
        verification_config: QA verification configuration
        config_path: Path to config.yml for loading API keys

    Returns:
        Tuple of (num_objects_processed, num_qa_verified, num_qa_passed)
    """
    logger.info(f"Processing JSON: {json_path.name}")
    logger.info(f"Using image: {image_path.name}")

    # Load the JSON file
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Parse into DocumentProcessResult
    result = DocumentProcessResult.from_dict(data)

    # Load images
    page_images, image, is_pdf = _load_document_images(image_path)
    if page_images is None and image is None:
        return 0, 0, 0

    # Create verifiers
    verifiers = _create_verifiers(verification_config, config_path=config_path)
    if not verifiers:
        logger.error("No verifiers created, skipping file")
        return 0, 0, 0

    # Process each object's QA pairs
    num_objects_processed = 0
    num_qa_verified = 0
    num_qa_passed = 0
    total_verifier_time = 0.0
    document_start_time = time.perf_counter()

    # Filter objects that have QA pairs
    valid_objects = {
        obj_id: obj
        for obj_id, obj in result.objects.items()
        if obj.qa and len(obj.qa) > 0
    }

    if not valid_objects:
        logger.warning(f"No objects with QA pairs found in {json_path.name}")
        return 0, 0, 0

    # Use tqdm to show progress
    progress_bar = tqdm(
        valid_objects.items(),
        desc=f"Verifying QA ({json_path.name[:30]})",
        unit="object",
        disable=len(valid_objects) <= 1,
        file=sys.stderr,
        dynamic_ncols=True,
    )

    for obj_id, obj in progress_bar:
        # Get appropriate image for this object
        obj_image = _get_object_image(obj, obj_id, is_pdf, page_images, image)
        if obj_image is None:
            continue

        has_verified_qa = False  # Initialize for each object
        for qa_pair in obj.qa:
            qa_start_time = time.perf_counter()
            try:
                verification_results = _verify_qa_pair(
                    qa_pair, verifiers, result.context, obj_image
                )

                if verification_results:
                    # Merge new verification results with existing ones
                    if qa_pair.verification is None:
                        qa_pair.verification = {}
                    qa_pair.verification.update(verification_results)
                    num_qa_verified += 1
                    has_verified_qa = True

                    # Check if all verifiers passed
                    all_passed = all(
                        v.get("response", "").lower() == "yes"
                        for v in verification_results.values()
                    )
                    if all_passed:
                        num_qa_passed += 1

            except Exception as e:
                logger.error(f"Failed to verify QA pair in object {obj_id}: {e}")
                continue
            finally:
                total_verifier_time += time.perf_counter() - qa_start_time

        if has_verified_qa:
            num_objects_processed += 1

        # Update progress bar
        progress_bar.set_postfix(verified=num_qa_verified, passed=num_qa_passed)

    document_elapsed_time = time.perf_counter() - document_start_time
    avg_time_per_qa = (
        total_verifier_time / num_qa_verified if num_qa_verified > 0 else 0.0
    )

    # Add verify_metadata to result
    result.verify_metadata = RunMetadata(
        runner_name="verify_batch",
        timestamp=datetime.utcnow().isoformat() + "Z",
        latency=document_elapsed_time * 1000.0,  # Convert to milliseconds
        extra={
            "objects_processed": num_objects_processed,
            "qa_verified": num_qa_verified,
            "qa_passed": num_qa_passed,
            "average_time_per_qa": avg_time_per_qa,
            "total_verifier_time": total_verifier_time,
            "verifiers_used": _collect_verifier_metadata(verifiers),
        },
    )

    # Write back to JSON file
    output_data = result.to_dict()
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Processed {num_objects_processed} objects, "
        f"verified {num_qa_verified} QA pairs, "
        f"{num_qa_passed} passed all verifiers "
        f"(total time: {document_elapsed_time:.2f}s, "
        f"avg {avg_time_per_qa:.2f}s per QA)"
    )
    return num_objects_processed, num_qa_verified, num_qa_passed


def process_batch_verification(
    input_path: Path,
    verification_config: QAVerificationConfig,
    image_dirs: Optional[List[Path]] = None,
    config_path: str | None = None,
) -> Tuple[int, int, int, int]:
    """Process JSON files to verify QA pairs.

    Args:
        input_path: Path to JSON file or directory of JSON files
        verification_config: QA verification configuration
        image_dirs: List of directories to search for images (default: same dir as JSON)
        config_path: Path to config.yml for loading API keys

    Returns:
        Tuple of (num_files_processed, total_objects, total_qa_verified, total_qa_passed)
    """
    # Collect JSON files
    json_files: List[Path] = []
    if input_path.is_file():
        if input_path.suffix.lower() == ".json":
            json_files = [input_path]
        else:
            logger.error(f"Input file is not a JSON file: {input_path}")
            return 0, 0, 0, 0
    elif input_path.is_dir():
        json_files = sorted(input_path.glob("*.json"))
    else:
        raise ValueError(f"Input path does not exist: {input_path}")

    if not json_files:
        logger.warning(f"No JSON files found in {input_path}")
        return 0, 0, 0, 0

    logger.info(f"Found {len(json_files)} JSON files to process")

    # Default image search directories
    if image_dirs is None:
        image_dirs = [input_path.parent if input_path.is_file() else input_path]

    num_files_processed = 0
    total_objects_processed = 0
    total_qa_verified = 0
    total_qa_passed = 0
    batch_start_time = time.perf_counter()

    for json_file in json_files:
        try:
            # Find corresponding image file
            image_path = find_image_for_json(json_file, image_dirs)
            if not image_path:
                logger.warning(f"Skipping {json_file.name}: Image not found")
                continue

            num_objects, num_qa, num_passed = process_document_verification(
                json_path=json_file,
                image_path=image_path,
                verification_config=verification_config,
                config_path=config_path,
            )

            num_files_processed += 1
            total_objects_processed += num_objects
            total_qa_verified += num_qa
            total_qa_passed += num_passed

        except Exception as e:
            logger.error(f"Failed to process {json_file}: {e}")
            continue

    batch_elapsed_time = time.perf_counter() - batch_start_time
    avg_time_per_qa = (
        batch_elapsed_time / total_qa_verified if total_qa_verified > 0 else 0.0
    )

    logger.info(
        f"Batch verification complete: {num_files_processed} files, "
        f"{total_objects_processed} objects, {total_qa_verified} QA pairs verified, "
        f"{total_qa_passed} passed all verifiers "
        f"(total time: {batch_elapsed_time:.2f}s, "
        f"avg {avg_time_per_qa:.2f}s per QA)"
    )

    return (
        num_files_processed,
        total_objects_processed,
        total_qa_verified,
        total_qa_passed,
    )


def clean_document_verification(json_path: Path) -> Tuple[int, int]:
    """Remove verification results from a single JSON document.

    Args:
        json_path: Path to the JSON file to clean

    Returns:
        Tuple of (qa_pairs_modified, verification_entries_removed)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = DocumentProcessResult.from_dict(data)

    qa_pairs_modified = 0
    verification_entries_removed = 0

    def _clear_verification(
        qas: List[QAPair], *, count: bool = True
    ) -> Tuple[int, int]:
        pairs = 0
        entries = 0
        for qa_pair in qas:
            if qa_pair.verification:
                if count:
                    entries += len(qa_pair.verification)
                    pairs += 1
                qa_pair.verification = None
        return pairs, entries

    for obj in result.objects.values():
        pairs, entries = _clear_verification(obj.qa)
        qa_pairs_modified += pairs
        verification_entries_removed += entries

    for obj in result.object_list:
        _clear_verification(obj.qa, count=False)

    if qa_pairs_modified > 0:
        output_data = result.to_dict()
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info(
            f"Removed verification results from {qa_pairs_modified} QA pairs in {json_path.name}"
        )
    else:
        logger.info(f"No verification results found in {json_path.name}")

    return qa_pairs_modified, verification_entries_removed


def clean_batch_verification(json_files: List[Path]) -> Tuple[int, int, int]:
    """Clean verification results for multiple JSON files.

    Args:
        json_files: List of JSON file paths to clean

    Returns:
        Tuple of (files_processed, total_qa_pairs_modified, total_entries_removed)
    """
    files_processed = 0
    total_pairs_modified = 0
    total_entries_removed = 0

    for json_path in json_files:
        if not json_path.exists():
            logger.warning(f"Skipping missing JSON file: {json_path}")
            continue

        pairs_modified, entries_removed = clean_document_verification(json_path)
        files_processed += 1
        total_pairs_modified += pairs_modified
        total_entries_removed += entries_removed

    return files_processed, total_pairs_modified, total_entries_removed
