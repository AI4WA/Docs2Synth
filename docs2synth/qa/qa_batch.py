"""Batch QA generation for preprocessed documents.

This module processes image files (or directories of images) to generate QA pairs.
It finds the corresponding preprocessed JSON files and generates questions for each object
using all strategies configured in config.yml.
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
from docs2synth.qa import QAGeneratorFactory
from docs2synth.qa.config import QAGenerationConfig, QAStrategyConfig
from docs2synth.utils.pdf_images import get_pdf_images
from docs2synth.utils.text import truncate_context

logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf", ".tiff", ".tif", ".bmp"}


def find_json_for_image(
    image_path: Path, output_dir: Path, processor_name: str = "paddleocr"
) -> Optional[Path]:
    """Find the corresponding preprocessed JSON file for an image.

    Args:
        image_path: Path to the image file
        output_dir: Directory containing preprocessed JSON files
        processor_name: Name of the processor used (e.g., "paddleocr")

    Returns:
        Path to the JSON file if found, None otherwise
    """
    # Expected JSON filename: {image_stem}_{processor}.json
    json_name = f"{image_path.stem}_{processor_name}.json"
    json_path = output_dir / json_name

    if json_path.exists():
        return json_path

    # Try without processor suffix
    json_name = f"{image_path.stem}.json"
    json_path = output_dir / json_name
    if json_path.exists():
        return json_path

    logger.warning(f"Could not find JSON file for image {image_path}")
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
        # For PDF files, load pre-converted page images (from preprocessing stage)
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
                f"Please run preprocessing first to convert PDF to images. "
                f"Expected images in: {image_path.parent / image_path.stem}/"
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


def _group_strategies_by_id(
    strategies: List[QAStrategyConfig],
) -> Dict[Optional[str], Dict[str, List[QAStrategyConfig]]]:
    """Group strategies by group_id and strategy type.

    Args:
        strategies: List of QA strategy configurations

    Returns:
        Dictionary mapping group_id to strategy type to list of configs
    """
    groups: Dict[Optional[str], Dict[str, List[QAStrategyConfig]]] = {}

    for strategy_config in strategies:
        group_id = strategy_config.group_id
        if group_id not in groups:
            groups[group_id] = {"semantic": [], "layout_aware": [], "logical_aware": []}

        strategy_name = strategy_config.strategy
        if strategy_name in groups[group_id]:
            groups[group_id][strategy_name].append(strategy_config)

    return groups


def _create_semantic_generator(
    config: QAStrategyConfig,
    group_id: Optional[str],
    config_path: Optional[str] = None,
) -> Optional[Tuple[Any, QAStrategyConfig]]:
    """Create a semantic generator from config.

    Args:
        config: QA strategy configuration
        group_id: Group ID for logging
        config_path: Path to config.yml for agent initialization

    Returns:
        Tuple of (generator, config) or None if creation fails
    """
    try:
        effective_config_path = config_path or "./config.yml"
        generator = QAGeneratorFactory.create_from_config(
            config, config_path=effective_config_path
        )
        logger.info(
            f"Group {group_id or 'default'}: Created semantic generator "
            f"({config.provider}/{config.model or 'default'})"
        )
        return (generator, config)
    except Exception as e:
        logger.error(
            f"Group {group_id or 'default'}: Failed to create semantic generator: {e}"
        )
        return None


def _create_transform_generators(
    configs: List[QAStrategyConfig],
    strategy_type: str,
    group_id: Optional[str],
    config_path: Optional[str] = None,
) -> List[Tuple[Any, QAStrategyConfig]]:
    """Create transform generators (layout_aware or logical_aware) from configs.

    Args:
        configs: List of QA strategy configurations
        strategy_type: Type of strategy ('layout_aware' or 'logical_aware')
        group_id: Group ID for logging
        config_path: Path to config.yml for agent initialization

    Returns:
        List of (generator, config) tuples
    """
    generators = []
    effective_config_path = config_path or "./config.yml"
    for config in configs:
        try:
            generator = QAGeneratorFactory.create_from_config(
                config, config_path=effective_config_path
            )
            generators.append((generator, config))
            logger.info(
                f"Group {group_id or 'default'}: Created {strategy_type} generator "
                f"({config.provider}/{config.model or 'default'})"
            )
        except Exception as e:
            logger.error(
                f"Group {group_id or 'default'}: Failed to create {strategy_type} generator: {e}"
            )
    return generators


def _create_qa_generators(
    qa_config: QAGenerationConfig,
    config_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Create QA generators from config, grouped by group_id.

    Args:
        qa_config: QA generation configuration
        config_path: Path to config.yml for agent initialization

    Returns:
        List of generator groups, each containing:
        {
            'group_id': str or None,
            'semantic': (generator, config) or None,
            'layout_aware': (generator, config) or None,
            'logical_aware': (generator, config) or None
        }
    """
    if not qa_config.strategies:
        logger.warning("No QA strategies configured in config.yml")
        return []

    groups = _group_strategies_by_id(qa_config.strategies)

    # Create generators for each group
    generator_groups = []
    for group_id, strategies_by_type in groups.items():
        group_info: Dict[str, Any] = {"group_id": group_id}

        # Create semantic generator (use first config if multiple)
        if strategies_by_type["semantic"]:
            semantic_result = _create_semantic_generator(
                strategies_by_type["semantic"][0], group_id, config_path
            )
            if semantic_result is None:
                continue  # Skip this group if semantic generator fails
            group_info["semantic"] = semantic_result

        # Create layout_aware generators (all configs in the group)
        if strategies_by_type["layout_aware"]:
            layout_generators = _create_transform_generators(
                strategies_by_type["layout_aware"],
                "layout_aware",
                group_id,
                config_path,
            )
            if layout_generators:
                group_info["layout_aware"] = layout_generators

        # Create logical_aware generators (all configs in the group)
        if strategies_by_type["logical_aware"]:
            logical_generators = _create_transform_generators(
                strategies_by_type["logical_aware"],
                "logical_aware",
                group_id,
                config_path,
            )
            if logical_generators:
                group_info["logical_aware"] = logical_generators

        # Only add groups that have at least a semantic generator
        if "semantic" in group_info:
            generator_groups.append(group_info)

    logger.info(f"Created {len(generator_groups)} generator group(s)")
    return generator_groups


def _collect_strategy_metadata(generator_groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collect lightweight metadata about configured QA strategies for serialization."""
    strategies: List[Dict[str, Any]] = []

    for group in generator_groups:
        group_id = group.get("group_id")

        if "semantic" in group:
            _, semantic_config = group["semantic"]
            strategies.append(
                {
                    "group_id": group_id,
                    "strategy": "semantic",
                    "provider": getattr(semantic_config, "provider", None),
                    "model": getattr(semantic_config, "model", None),
                }
            )

        for strategy_type in ("layout_aware", "logical_aware"):
            for _, strategy_config in group.get(strategy_type, []):
                strategies.append(
                    {
                        "group_id": group_id,
                        "strategy": strategy_type,
                        "provider": getattr(strategy_config, "provider", None),
                        "model": getattr(strategy_config, "model", None),
                    }
                )

    return strategies


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
        # Fallback: should not happen, but handle gracefully
        logger.error(
            f"Could not determine image for object {obj_id} "
            f"(is_pdf={is_pdf}, page_images={page_images is not None}, image={image is not None})"
        )
        return None


def _generate_semantic_questions(
    obj: Any,
    obj_id: int,
    obj_image: PILImage.Image,
    context: str,
    semantic_generators: List[Tuple[str, Any, Any]],
    config: Optional[Any] = None,
) -> Tuple[Optional[str], int]:
    """Generate semantic questions for an object.

    Args:
        obj: DocumentObject instance
        obj_id: Object ID for logging
        obj_image: Image for this object
        context: Document context
        semantic_generators: List of (strategy, generator, config) tuples
        config: Config object to read max_model_len for vLLM (optional)

    Returns:
        Tuple of (first semantic question generated, number of questions generated)
    """
    semantic_question = None
    num_generated = 0

    # Truncate context if too long (use first strategy config for model info)
    strategy_config_for_truncation = (
        semantic_generators[0][2] if semantic_generators else None
    )

    # Get max_model_len from config for vLLM
    max_model_len = None
    if (
        config
        and strategy_config_for_truncation
        and strategy_config_for_truncation.provider
    ):
        if "vllm" in strategy_config_for_truncation.provider.lower():
            max_model_len = config.get("agent.vllm.max_model_len")

    truncated_context, was_truncated = truncate_context(
        context,
        max_tokens=(
            strategy_config_for_truncation.max_tokens
            if strategy_config_for_truncation
            else None
        ),
        provider=(
            strategy_config_for_truncation.provider
            if strategy_config_for_truncation
            else None
        ),
        model=(
            strategy_config_for_truncation.model
            if strategy_config_for_truncation
            else None
        ),
        max_model_len=max_model_len,
    )
    if was_truncated:
        logger.warning(
            f"Context truncated for object {obj_id} "
            f"(original length: {len(context)} chars, truncated to: {len(truncated_context)} chars)"
        )

    for strategy, generator, strategy_config in semantic_generators:
        try:
            image_available = obj_image is not None
            logger.info(
                f"[semantic/{strategy}] object {obj_id}: image_provided={image_available}"
            )
            # Generate semantic question from context and target
            question = generator.generate(
                context=truncated_context,
                target=obj.text,
                image=obj_image,
                temperature=strategy_config.temperature,
                max_tokens=strategy_config.max_tokens,
            )

            # Use first semantic question as base for transformation strategies
            if semantic_question is None:
                semantic_question = question.strip()

            # Create QA pair
            # Answer is always the object's text (for semantic questions)
            qa_pair = QAPair(
                question=question.strip(),
                answer=obj.text,
                strategy=strategy,
                extra={
                    "provider": strategy_config.provider,
                    "model": strategy_config.model or "default",
                },
            )

            # Add to object's qa list
            obj.qa.append(qa_pair)
            num_generated += 1

            logger.debug(
                f"Generated question for object {obj_id} ({strategy}): {question[:50]}..."
            )

        except Exception as e:
            logger.error(
                f"Failed to generate question for object {obj_id} with strategy {strategy}: {e}"
            )
            continue

    return semantic_question, num_generated


def _generate_transform_questions(
    obj: Any,
    obj_id: int,
    obj_image: PILImage.Image,
    semantic_question: str,
    transform_generators: List[Tuple[str, Any, Any]],
) -> int:
    """Generate transformed questions (layout_aware, logical_aware) for an object.

    Args:
        obj: DocumentObject instance
        obj_id: Object ID for logging
        obj_image: Image for this object
        semantic_question: Base semantic question to transform
        transform_generators: List of (strategy, generator, config) tuples

    Returns:
        Number of questions generated
    """
    num_generated = 0

    for strategy, generator, strategy_config in transform_generators:
        try:
            image_available = obj_image is not None
            logger.info(
                f"[transform/{strategy}] object {obj_id}: image_provided={image_available}"
            )
            # Transform the semantic question
            transformed_question = generator.generate(
                question=semantic_question,
                image=obj_image,
                temperature=strategy_config.temperature,
                max_tokens=strategy_config.max_tokens,
            )

            # Create QA pair
            # Answer is always the object's text (same as semantic question)
            qa_pair = QAPair(
                question=transformed_question.strip(),
                answer=obj.text,
                strategy=strategy,
                extra={
                    "provider": strategy_config.provider,
                    "model": strategy_config.model or "default",
                    "original_question": semantic_question,
                },
            )

            # Add to object's qa list
            obj.qa.append(qa_pair)
            num_generated += 1

            logger.debug(
                f"Generated transformed question for object {obj_id} ({strategy}): {transformed_question[:50]}..."
            )

        except Exception as e:
            logger.error(
                f"Failed to transform question for object {obj_id} with strategy {strategy}: {e}"
            )
            continue

    return num_generated


def _generate_semantic_question_for_group(
    obj: Any,
    obj_id: int,
    obj_image: PILImage.Image,
    context: str,
    group: Dict[str, Any],
    config: Optional[Any] = None,
) -> Optional[str]:
    """Generate semantic question for an object using a generator group.

    Args:
        obj: DocumentObject instance
        obj_id: Object ID for logging
        obj_image: Image for this object
        context: Document context
        group: Generator group dictionary
        config: Config object to read max_model_len for vLLM (optional)

    Returns:
        Semantic question string or None if generation fails
    """
    if "semantic" not in group:
        return None

    group_id = group.get("group_id")
    semantic_generator, semantic_config = group["semantic"]

    # Get max_model_len from config for vLLM
    max_model_len = None
    if (
        config
        and semantic_config.provider
        and "vllm" in semantic_config.provider.lower()
    ):
        max_model_len = config.get("agent.vllm.max_model_len")

    # Truncate context if too long
    truncated_context, was_truncated = truncate_context(
        context,
        max_tokens=semantic_config.max_tokens,
        provider=semantic_config.provider,
        model=semantic_config.model,
        max_model_len=max_model_len,
    )
    if was_truncated:
        logger.warning(
            f"Context truncated for object {obj_id} (group: {group_id or 'default'}, "
            f"original length: {len(context)} chars, truncated to: {len(truncated_context)} chars)"
        )

    try:
        # Pass max_model_len via kwargs so semantic generator can use it for truncation if needed
        gen_kwargs = {}
        if max_model_len is not None:
            gen_kwargs["max_model_len"] = max_model_len

        question = semantic_generator.generate(
            context=truncated_context,
            target=obj.text,
            image=obj_image,
            temperature=semantic_config.temperature,
            max_tokens=semantic_config.max_tokens,
            **gen_kwargs,
        )
        semantic_question = question.strip()

        # Create QA pair for semantic question
        qa_pair = QAPair(
            question=semantic_question,
            answer=obj.text,
            strategy="semantic",
            extra={
                "provider": semantic_config.provider,
                "model": semantic_config.model or "default",
                "group_id": group_id,
            },
        )
        obj.qa.append(qa_pair)

        logger.debug(
            f"Generated semantic question for object {obj_id} "
            f"(group: {group_id or 'default'}): {question[:50]}..."
        )
        return semantic_question
    except Exception as e:
        logger.error(
            f"Failed to generate semantic question for object {obj_id} "
            f"(group: {group_id or 'default'}): {e}"
        )
        return None


def _generate_transform_questions_for_group(
    obj: Any,
    obj_id: int,
    obj_image: PILImage.Image,
    semantic_question: str,
    group: Dict[str, Any],
) -> int:
    """Generate transform questions (layout_aware, logical_aware) for an object.

    Args:
        obj: DocumentObject instance
        obj_id: Object ID for logging
        obj_image: Image for this object
        semantic_question: Base semantic question to transform
        group: Generator group dictionary

    Returns:
        Number of questions generated
    """
    group_id = group.get("group_id")
    num_generated = 0

    # Process layout_aware generators
    if "layout_aware" in group:
        for layout_generator, layout_config in group["layout_aware"]:
            layout_provider = layout_config.provider
            layout_model = layout_config.model or "default"

            # Check if layout_aware question already exists for this exact combination
            if _has_existing_qa_for_group(
                obj, group_id, "layout_aware", layout_provider, layout_config.model
            ):
                logger.info(
                    f"Skipping layout_aware for object {obj_id} "
                    f"(group: '{group_id or 'default'}', provider: '{layout_provider}', "
                    f"model: '{layout_model}'): already exists"
                )
                continue

            try:
                transformed_question = layout_generator.generate(
                    question=semantic_question,
                    image=obj_image,
                    temperature=layout_config.temperature,
                    max_tokens=layout_config.max_tokens,
                )

                qa_pair = QAPair(
                    question=transformed_question.strip(),
                    answer=obj.text,
                    strategy="layout_aware",
                    extra={
                        "provider": layout_config.provider,
                        "model": layout_config.model or "default",
                        "original_question": semantic_question,
                        "group_id": group_id,
                    },
                )
                obj.qa.append(qa_pair)
                num_generated += 1

                logger.debug(
                    f"Generated layout_aware question for object {obj_id} "
                    f"(group: {group_id or 'default'}): {transformed_question[:50]}..."
                )
            except Exception as e:
                logger.error(
                    f"Failed to generate layout_aware question for object {obj_id} "
                    f"(group: {group_id or 'default'}): {e}"
                )

    # Process logical_aware generators
    if "logical_aware" in group:
        for logical_generator, logical_config in group["logical_aware"]:
            logical_provider = logical_config.provider
            logical_model = logical_config.model or "default"

            # Check if logical_aware question already exists for this exact combination
            if _has_existing_qa_for_group(
                obj, group_id, "logical_aware", logical_provider, logical_config.model
            ):
                logger.info(
                    f"Skipping logical_aware for object {obj_id} "
                    f"(group: '{group_id or 'default'}', provider: '{logical_provider}', "
                    f"model: '{logical_model}'): already exists"
                )
                continue

            try:
                transformed_question = logical_generator.generate(
                    question=semantic_question,
                    image=obj_image,
                    temperature=logical_config.temperature,
                    max_tokens=logical_config.max_tokens,
                )

                qa_pair = QAPair(
                    question=transformed_question.strip(),
                    answer=obj.text,
                    strategy="logical_aware",
                    extra={
                        "provider": logical_config.provider,
                        "model": logical_config.model or "default",
                        "original_question": semantic_question,
                        "group_id": group_id,
                    },
                )
                obj.qa.append(qa_pair)
                num_generated += 1

                logger.debug(
                    f"Generated logical_aware question for object {obj_id} "
                    f"(group: {group_id or 'default'}): {transformed_question[:50]}..."
                )
            except Exception as e:
                logger.error(
                    f"Failed to generate logical_aware question for object {obj_id} "
                    f"(group: {group_id or 'default'}): {e}"
                )

    return num_generated


def _has_existing_qa_for_group(
    obj: Any,
    group_id: Optional[str],
    strategy: str,
    provider: str,
    model: Optional[str],
) -> bool:
    """Check if object already has QA pairs for a given strategy + provider + model + group_id combination.

    Args:
        obj: DocumentObject instance
        group_id: Group ID to check (None means default group)
        strategy: Strategy type to check (e.g., "semantic", "layout_aware", "logical_aware")
        provider: Provider name to check
        model: Model name to check (None means "default")

    Returns:
        True if a QA pair exists with matching strategy + provider + model + group_id, False otherwise
    """
    if not obj.qa:
        return False

    # Normalize model: None or empty string means "default"
    expected_model = model or "default"

    # Debug: log all existing QA pairs for this strategy
    existing_qa_for_strategy = [
        (
            qa.strategy,
            qa.extra.get("provider") if qa.extra else None,
            qa.extra.get("model") if qa.extra else None,
            qa.extra.get("group_id") if qa.extra else None,
        )
        for qa in obj.qa
        if qa.strategy == strategy
    ]
    if existing_qa_for_strategy:
        logger.debug(
            f"Existing QA pairs for strategy '{strategy}': {existing_qa_for_strategy}, "
            f"looking for provider='{provider}', model='{expected_model}', group_id='{group_id}'"
        )

    for qa_pair in obj.qa:
        if qa_pair.strategy == strategy:
            # Check if provider, model, and group_id all match
            # These fields are stored in extra dict (from JSON parsing)
            qa_extra = qa_pair.extra if qa_pair.extra else {}
            qa_provider = qa_extra.get("provider")
            qa_model = (
                qa_extra.get("model") or "default"
            )  # Normalize None/empty to "default"
            qa_group_id = qa_extra.get("group_id")

            # Check all four fields match
            if (
                qa_provider == provider
                and qa_model == expected_model
                and qa_group_id == group_id
            ):
                logger.debug(
                    f"Found existing QA pair: strategy='{strategy}', "
                    f"provider='{qa_provider}', model='{qa_model}', group_id='{qa_group_id}' "
                    f"(looking for provider='{provider}', model='{expected_model}', group_id='{group_id}')"
                )
                return True
    return False


def _process_object_with_groups(
    obj_id: int,
    obj: Any,
    obj_image: PILImage.Image,
    context: str,
    generator_groups: List[Dict[str, Any]],
    config: Optional[Any] = None,
) -> Tuple[int, float]:
    """Process a single object across all generator groups.

    Args:
        obj_id: Object ID for logging
        obj: DocumentObject instance
        obj_image: Image for this object
        context: Document context
        generator_groups: List of generator groups
        config: Config object to read max_model_len for vLLM (optional)

    Returns:
        Tuple of (number of questions generated, total time taken in seconds)
    """
    num_questions_generated = 0
    start_time = time.perf_counter()

    for group in generator_groups:
        group_id = group.get("group_id")

        # Get semantic config to check provider and model
        if "semantic" not in group:
            continue

        semantic_generator, semantic_config = group["semantic"]
        semantic_provider = semantic_config.provider
        semantic_model = semantic_config.model or "default"

        # Check if semantic question already exists for this exact combination
        has_existing = _has_existing_qa_for_group(
            obj, group_id, "semantic", semantic_provider, semantic_config.model
        )
        if has_existing:
            logger.info(
                f"Skipping semantic for object {obj_id} "
                f"(group: '{group_id or 'default'}', provider: '{semantic_provider}', "
                f"model: '{semantic_model}'): already exists"
            )
            continue

        # Stage 1: Generate semantic question for this group
        logger.debug(
            f"Generating semantic question for object {obj_id} "
            f"(group: '{group_id or 'default'}', provider: '{semantic_provider}', "
            f"model: '{semantic_model}')"
        )
        semantic_question = _generate_semantic_question_for_group(
            obj, obj_id, obj_image, context, group, config
        )
        if semantic_question is None:
            continue  # Skip transform strategies if semantic fails

        num_questions_generated += 1  # Count semantic question

        # Stage 2: Transform semantic question using layout_aware and logical_aware strategies
        # Check each transform strategy individually before generating
        num_questions_generated += _generate_transform_questions_for_group(
            obj, obj_id, obj_image, semantic_question, group
        )

    elapsed_time = time.perf_counter() - start_time
    return num_questions_generated, elapsed_time


def process_document(
    image_path: Path,
    json_path: Path,
    qa_config: QAGenerationConfig,
    config: Optional[Any] = None,
    config_path: Optional[str] = None,
) -> Tuple[int, int]:
    """Process a single document to generate QA pairs.

    Args:
        image_path: Path to the image file
        json_path: Path to the preprocessed JSON file
        qa_config: QA generation configuration from config.yml
        config: Config object to read max_model_len for vLLM (optional)
        config_path: Path to config.yml for agent initialization (optional)

    Returns:
        Tuple of (num_objects_processed, num_questions_generated)
    """
    logger.info(f"Processing image: {image_path.name}")
    logger.info(f"Using JSON: {json_path.name}")

    # Load the JSON file
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Parse into DocumentProcessResult
    result = DocumentProcessResult.from_dict(data)

    # Load images
    page_images, image, is_pdf = _load_document_images(image_path)
    if page_images is None and image is None:
        return 0, 0

    # Create generator groups
    generator_groups = _create_qa_generators(qa_config, config_path=config_path)
    if not generator_groups:
        logger.error("No generator groups created, skipping file")
        return 0, 0

    # Process each object
    num_objects_processed = 0
    num_questions_generated = 0
    total_question_time = 0.0
    document_start_time = time.perf_counter()

    # Filter out empty objects for progress tracking
    valid_objects = {
        obj_id: obj for obj_id, obj in result.objects.items() if obj.text.strip()
    }

    # Use tqdm to show progress bar for objects
    progress_bar = tqdm(
        valid_objects.items(),
        desc=f"Generating QA ({image_path.name[:30]})",
        unit="object",
        disable=len(valid_objects)
        <= 1,  # Disable for single object to keep output clean
        file=sys.stderr,  # Use stderr to avoid conflicts with stdout
        dynamic_ncols=True,  # Adjust width based on terminal
    )

    for obj_id, obj in progress_bar:
        # Update progress bar with current object info
        progress_bar.set_postfix(
            text=obj.text[:20] + "..." if len(obj.text) > 20 else obj.text
        )
        progress_bar.refresh()  # Force refresh before processing

        logger.debug(f"Processing object {obj_id}: {obj.text[:50]}...")

        # Get appropriate image for this object
        obj_image = _get_object_image(obj, obj_id, is_pdf, page_images, image)
        if obj_image is None:
            continue

        # Process object with all generator groups
        questions_generated, question_time = _process_object_with_groups(
            obj_id, obj, obj_image, result.context, generator_groups, config
        )
        num_questions_generated += questions_generated
        total_question_time += question_time

        if obj.qa:
            num_objects_processed += 1

    document_elapsed_time = time.perf_counter() - document_start_time
    avg_time_per_question = (
        total_question_time / num_questions_generated
        if num_questions_generated > 0
        else 0.0
    )

    result.qa_metadata = RunMetadata(
        runner_name="qa_batch",
        timestamp=datetime.utcnow().isoformat() + "Z",
        latency=document_elapsed_time * 1000.0,  # Convert to milliseconds
        extra={
            "objects_processed": num_objects_processed,
            "questions_generated": num_questions_generated,
            "average_time_per_question": avg_time_per_question,
            "total_question_time": total_question_time,
            "strategies_used": _collect_strategy_metadata(generator_groups),
        },
    )

    # Write back to JSON file
    output_data = result.to_dict()
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Processed {num_objects_processed} objects, generated {num_questions_generated} questions "
        f"(total time: {document_elapsed_time:.2f}s, "
        f"avg {avg_time_per_question:.2f}s per question)"
    )
    return num_objects_processed, num_questions_generated


def process_batch(
    input_path: Path,
    output_dir: Path,
    qa_config: QAGenerationConfig,
    processor_name: str = "paddleocr",
    config: Optional[Any] = None,
    config_path: Optional[str] = None,
) -> Tuple[int, int, int]:
    """Process image files to generate QA pairs.

    Args:
        input_path: Path to image file or directory of images
        output_dir: Directory containing preprocessed JSON files
        qa_config: QA generation configuration from config.yml
        processor_name: Name of the processor used (for finding JSON files)
        config: Config object to read max_model_len for vLLM (optional)
        config_path: Path to config.yml for agent initialization (optional)

    Returns:
        Tuple of (num_files_processed, total_objects_processed, total_questions_generated)
    """
    # Collect image files
    image_files: List[Path] = []
    if input_path.is_file():
        if input_path.suffix.lower() in IMAGE_EXTENSIONS:
            image_files = [input_path]
        else:
            logger.error(f"Input file is not a supported image format: {input_path}")
            return 0, 0, 0
    elif input_path.is_dir():
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(input_path.glob(f"*{ext}"))
        image_files = sorted(image_files)
    else:
        raise ValueError(f"Input path does not exist: {input_path}")

    if not image_files:
        logger.warning(f"No image files found in {input_path}")
        return 0, 0, 0

    logger.info(f"Found {len(image_files)} image files to process")

    num_files_processed = 0
    total_objects_processed = 0
    total_questions_generated = 0
    batch_start_time = time.perf_counter()

    for image_file in image_files:
        try:
            # Find corresponding JSON file
            json_path = find_json_for_image(image_file, output_dir, processor_name)
            if not json_path:
                logger.warning(f"Skipping {image_file.name}: JSON not found")
                continue

            num_objects, num_questions = process_document(
                image_path=image_file,
                json_path=json_path,
                qa_config=qa_config,
                config=config,
                config_path=config_path,
            )

            num_files_processed += 1
            total_objects_processed += num_objects
            total_questions_generated += num_questions

        except Exception as e:
            logger.error(f"Failed to process {image_file}: {e}")
            continue

    batch_elapsed_time = time.perf_counter() - batch_start_time
    avg_time_per_question = (
        batch_elapsed_time / total_questions_generated
        if total_questions_generated > 0
        else 0.0
    )

    logger.info(
        f"Batch processing complete: {num_files_processed} files, "
        f"{total_objects_processed} objects, {total_questions_generated} questions "
        f"(total time: {batch_elapsed_time:.2f}s, "
        f"avg {avg_time_per_question:.2f}s per question)"
    )

    return num_files_processed, total_objects_processed, total_questions_generated


def clean_document_qa(json_path: Path) -> Tuple[int, int]:
    """Remove QA pairs from a single JSON document.

    Args:
        json_path: Path to the JSON file to clean

    Returns:
        Tuple of (objects_modified, qa_pairs_removed)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = DocumentProcessResult.from_dict(data)

    objects_modified = 0
    qa_pairs_removed = 0

    for obj in result.objects.values():
        if obj.qa:
            qa_pairs_removed += len(obj.qa)
            obj.qa = []
            objects_modified += 1

    for obj in result.object_list:
        if obj.qa:
            obj.qa = []

    if qa_pairs_removed > 0:
        output_data = result.to_dict()
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info(
            f"Removed {qa_pairs_removed} QA pairs from {json_path.name} ({objects_modified} objects)"
        )
    else:
        logger.info(f"No QA pairs found in {json_path.name}")

    return objects_modified, qa_pairs_removed


def clean_batch_qa(json_files: List[Path]) -> Tuple[int, int, int]:
    """Clean QA pairs for multiple JSON files.

    Args:
        json_files: List of JSON file paths to clean

    Returns:
        Tuple of (files_processed, total_objects_modified, total_qa_removed)
    """
    files_processed = 0
    total_objects_modified = 0
    total_qa_removed = 0

    for json_path in json_files:
        if not json_path.exists():
            logger.warning(f"Skipping missing JSON file: {json_path}")
            continue

        objects_modified, qa_removed = clean_document_qa(json_path)
        files_processed += 1
        total_objects_modified += objects_modified
        total_qa_removed += qa_removed

    return files_processed, total_objects_modified, total_qa_removed
