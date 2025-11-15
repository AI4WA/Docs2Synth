"""Preprocessing utilities for converting JSON QA pairs to training tensors.

This module provides functions to preprocess verified QA pairs from JSON files
into the tensor format required by the training functions.

Stage 1 (MVP): Implements basic fields with simplified versions for complex features.
Stage 2: Will implement full feature extraction (visual features, BERT embeddings, etc.)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from docs2synth.utils.logging import get_logger

logger = get_logger(__name__)

# Default preprocessing constants
DEFAULT_MAX_LENGTH = 512
DEFAULT_NUM_OBJECTS = 50
DEFAULT_HIDDEN_SIZE = 768
DEFAULT_POSITIONAL_ENCODING_DIM = 128


class PreprocessedQADataset(Dataset):
    """Dataset that preprocesses QA pairs into training tensor format.

    This is Stage 1 (MVP) implementation with simplified features:
    - Basic fields: Uses LayoutLMv3 processor (input_ids, attention_mask, bbox, pixel_values)
    - Complex fields: Simplified or placeholder implementations

    Stage 2 will add full feature extraction.
    """

    def __init__(
        self,
        qa_pairs: List[Dict[str, Any]],
        processor: Any,
        image_dir: Path,
        max_length: int = DEFAULT_MAX_LENGTH,
        num_objects: int = DEFAULT_NUM_OBJECTS,
    ):
        """Initialize the dataset.

        Args:
            qa_pairs: List of verified QA pair dictionaries from load_verified_qa_pairs()
            processor: LayoutLMv3 AutoProcessor from transformers
            image_dir: Directory containing document images
            max_length: Maximum sequence length for tokenization (default: 512)
            num_objects: Maximum number of objects per document (default: 50)
        """
        self.qa_pairs = qa_pairs
        self.processor = processor
        self.image_dir = Path(image_dir)
        self.max_length = max_length
        self.num_objects = num_objects

        # Detect tokenizer capability once at initialization
        self._detect_tokenizer()

        logger.info(f"Initialized PreprocessedQADataset with {len(qa_pairs)} QA pairs")
        logger.info(f"Image directory: {self.image_dir}")
        logger.info(f"Max sequence length: {max_length}")
        logger.info(f"Max objects per document: {num_objects}")
        logger.info(f"Using tokenizer: {self.tokenizer_name}")

    def __len__(self) -> int:
        return len(self.qa_pairs)

    def _detect_tokenizer(self):
        """Detect which tokenizer to use and cache it.

        LayoutLMv3 tokenizer requires pre-tokenized input.
        We test once and fall back to BERT if needed.
        """
        try:
            # Try LayoutLMv3 tokenizer with raw text
            test_text = "test"
            self.processor.tokenizer.encode_plus(
                test_text,
                add_special_tokens=True,
                padding="max_length",
                truncation=True,
                max_length=10,
                return_tensors="pt",
            )
            # If no exception, LayoutLMv3 tokenizer works
            self.tokenizer = self.processor.tokenizer
            self.tokenizer_name = "LayoutLMv3"
            logger.info("LayoutLMv3 tokenizer supports raw text, using it directly")
        except Exception as e:
            # LayoutLMv3 tokenizer requires pre-tokenized input, use BERT fallback
            logger.info(
                f"LayoutLMv3 tokenizer requires pre-tokenized input ({e}), using BERT tokenizer fallback"
            )
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.tokenizer_name = "BERT (bert-base-uncased)"

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """Get a preprocessed example.

        Returns:
            Dictionary with all required tensor fields for training, or None if failed
        """
        qa_pair = self.qa_pairs[idx]

        try:
            result = self._preprocess(qa_pair)
            if result is None:
                logger.debug(f"Skipping QA pair {idx}: preprocessing returned None")
            return result
        except Exception as e:
            logger.error(f"Error preprocessing QA pair {idx}: {e}")
            logger.error(f"QA pair: {qa_pair.get('question', 'N/A')}")
            # Return None instead of raising to allow skipping bad samples
            return None

    def _preprocess(self, qa: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Preprocess a single QA pair into training format.

        Uses word-level labels and processor's word_labels parameter
        for correct answer span detection.

        Args:
            qa: QA pair dictionary with keys:
                - question, answer, object_text, context
                - bbox, json_file, etc.

        Returns:
            Dictionary with all required tensor fields, or None if answer not found
        """
        # 1. Load image
        image = self._load_image(qa)

        # 2. Load OCR data from JSON
        ocr_data = self._load_ocr_data(qa)
        if ocr_data is None:
            logger.warning(f"No OCR data found for {qa['json_file']}")
            return None

        # 3. Prepare words, boxes, and labels
        question_words = qa["question"].split()  # Simple word split for question
        doc_words = ocr_data["words"]
        doc_boxes = ocr_data["boxes"]

        # Create word-level labels: 0=normal, 1=answer_start, 2=answer_end
        labels = self._create_word_labels(doc_words, qa["answer"])
        if labels is None:
            # Answer not found in document words
            logger.debug(f"Answer '{qa['answer']}' not found in document words")
            return None

        # Combine question (with zero boxes) + document words and boxes
        all_words = question_words + doc_words
        all_boxes = [[0, 0, 0, 0]] * len(question_words) + doc_boxes
        all_labels = [0] * len(question_words) + labels

        # 4. Use processor with word_labels for correct tokenization alignment
        try:
            encoding = self.processor(
                images=image,
                text=all_words,
                boxes=all_boxes,
                word_labels=all_labels,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        except Exception as e:
            logger.warning(f"Processor failed: {e}, using fallback tokenizer")
            # Fallback: use tokenizer without boxes
            text = " ".join(all_words)
            encoding = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            # Manually process image
            image_encoding = self.processor.image_processor(
                images=image, return_tensors="pt"
            )
            encoding["pixel_values"] = image_encoding["pixel_values"]
            # Create dummy labels
            encoding["labels"] = torch.zeros_like(encoding["input_ids"])

        # 5. Extract start/end positions from encoded labels
        encoded_labels = encoding.get("labels", torch.zeros_like(encoding["input_ids"]))
        start_idx = (encoded_labels == 1).nonzero(as_tuple=True)[1].tolist()
        end_idx = (encoded_labels == 2).nonzero(as_tuple=True)[1].tolist()

        if len(start_idx) == 0 or len(end_idx) == 0:
            logger.debug(
                f"Answer span not found after encoding (question: {qa['question'][:50]}...)"
            )
            return None

        # 6. Build result dictionary
        # Get bbox from encoding or create zeros
        bbox_tensor = encoding.get(
            "bbox",
            torch.zeros_like(encoding["input_ids"].squeeze(0))
            .unsqueeze(-1)
            .repeat(1, 4),
        ).squeeze(0)

        # Ensure bbox is properly normalized to 0-1000 range for LayoutLMv3
        # LayoutLMv3 expects bbox coordinates in range [0, 1000]
        bbox_tensor = torch.clamp(bbox_tensor, 0, 1000).long()

        result = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "bbox": bbox_tensor,
            "start_id": torch.tensor(start_idx[0], dtype=torch.long),
            "end_id": torch.tensor(end_idx[0], dtype=torch.long),
        }

        # Validate input_ids shape
        input_ids = result["input_ids"]
        if input_ids is None or input_ids.numel() == 0:
            logger.warning(
                f"Empty input_ids for question: {qa.get('question', 'N/A')[:50]}"
            )
            return None

        seq_len = input_ids.size(0)
        if seq_len is None or seq_len == 0:
            logger.warning(
                f"Invalid seq_len ({seq_len}) for question: {qa.get('question', 'N/A')[:50]}"
            )
            return None

        # 7. Add additional fields (simplified for Stage 1)
        # Handle token_type_ids - may not exist or may have wrong shape
        token_type_ids = encoding.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.squeeze(0)
            # Ensure it has the correct length
            if token_type_ids.size(0) != seq_len:
                logger.debug(
                    f"token_type_ids length mismatch, creating zeros: {token_type_ids.size(0)} != {seq_len}"
                )
                token_type_ids = torch.zeros(seq_len, dtype=torch.long)
        else:
            token_type_ids = torch.zeros(seq_len, dtype=torch.long)

        result.update(
            {
                "token_type_ids": token_type_ids,
                "token_objt_ids": torch.zeros(seq_len, dtype=torch.long),
                "visual_feat": torch.zeros(
                    self.num_objects, DEFAULT_HIDDEN_SIZE, dtype=torch.float
                ),
                "bert_cls": torch.zeros(DEFAULT_HIDDEN_SIZE, dtype=torch.float),
                "positional_encoding": torch.zeros(
                    self.num_objects, DEFAULT_POSITIONAL_ENCODING_DIM, dtype=torch.float
                ),
                "norm_bbox": torch.zeros(self.num_objects, 4, dtype=torch.float),
                "object_mask": torch.ones(self.num_objects, dtype=torch.float),
            }
        )

        return result

    def _load_ocr_data(self, qa: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Load OCR words and bounding boxes from JSON file.

        Args:
            qa: QA pair dictionary with json_file path

        Returns:
            Dictionary with 'words' and 'boxes' lists (boxes normalized to 0-1000), or None if not found
        """
        import json

        json_path = Path(qa["json_file"])
        if not json_path.exists():
            return None

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract words and boxes from JSON structure
            # JSON has structure: {"objects": {"0": {"text": "...", "bbox": [...]}, ...}}
            words = []
            boxes = []

            # Get image dimensions for normalization
            # Handle both dict and string context formats
            context = data.get("context", {})
            if isinstance(context, dict):
                image_width = context.get("size", {}).get("width", 1000)
                image_height = context.get("size", {}).get("height", 1000)
            else:
                # Context is a string (actual OCR text), use default dimensions
                image_width = 1000
                image_height = 1000

            if "objects" in data:
                # Sort by object ID to maintain reading order
                object_ids = sorted(data["objects"].keys(), key=lambda x: int(x))
                for obj_id in object_ids:
                    obj = data["objects"][obj_id]
                    if "text" in obj and "bbox" in obj:
                        text = obj["text"]
                        bbox = obj["bbox"]

                        # Split text into words
                        text_words = text.split()
                        # Use the same bbox for all words from this object
                        # (Simplified - Stage 2 will do word-level bbox alignment)
                        for word in text_words:
                            words.append(word)
                            if len(bbox) == 4:
                                # Normalize bbox to 0-1000 range for LayoutLMv3
                                # bbox format is [x0, y0, x1, y1]
                                normalized_bbox = [
                                    int((bbox[0] / image_width) * 1000),
                                    int((bbox[1] / image_height) * 1000),
                                    int((bbox[2] / image_width) * 1000),
                                    int((bbox[3] / image_height) * 1000),
                                ]
                                # Clamp to valid range
                                normalized_bbox = [
                                    max(0, min(1000, coord))
                                    for coord in normalized_bbox
                                ]
                                boxes.append(normalized_bbox)
                            else:
                                boxes.append([0, 0, 0, 0])

            if not words:
                logger.debug(f"No words found in {json_path}")
                return None

            return {"words": words, "boxes": boxes}

        except Exception as e:
            logger.warning(f"Failed to load OCR data from {json_path}: {e}")
            return None

    def _create_word_labels(
        self, doc_words: List[str], answer: str
    ) -> Optional[List[int]]:
        """Create word-level labels for answer span detection.

        Args:
            doc_words: List of document words
            answer: Answer string to find

        Returns:
            List of labels (0=normal, 1=start, 2=end), or None if answer not found
        """
        if not answer or not answer.strip():
            # Empty answer
            return None

        answer_words = answer.strip().split()
        if not answer_words:
            return None

        # Try to find answer words in document words
        answer_len = len(answer_words)

        for i in range(len(doc_words) - answer_len + 1):
            # Check if this is a match (case-insensitive)
            match = True
            for j, ans_word in enumerate(answer_words):
                if doc_words[i + j].lower() != ans_word.lower():
                    match = False
                    break

            if match:
                # Found answer! Create labels
                labels = [0] * len(doc_words)
                labels[i] = 1  # Start token
                labels[i + answer_len - 1] = 2  # End token
                return labels

        # Try fuzzy matching: find substring matches
        answer_text_lower = answer.lower().replace(" ", "")
        for i, word in enumerate(doc_words):
            word_lower = word.lower().replace(" ", "")
            if answer_text_lower in word_lower or word_lower in answer_text_lower:
                # Single word match
                labels = [0] * len(doc_words)
                labels[i] = 1  # Start
                labels[i] = 2  # End (same position for single word)
                return labels

        # Answer not found
        return None

    def _load_image(self, qa: Dict[str, Any]) -> Image.Image:
        """Load document image from QA pair metadata.

        Args:
            qa: QA pair dictionary

        Returns:
            PIL Image object
        """
        # Extract image filename from json_file path
        # Example: "data/processed/dev/cord_train_image_0159_docling.json"
        # -> "cord_train_image_0159.png"
        json_file = qa["json_file"]
        json_path = Path(json_file)

        # Remove processor suffix (_docling, _paddleocr, etc.)
        stem = json_path.stem
        if "_" in stem:
            # Try to remove processor suffix
            parts = stem.rsplit("_", 1)
            if parts[-1] in ["docling", "paddleocr", "easyocr", "pdfplumber"]:
                image_filename = parts[0] + ".png"
            else:
                image_filename = stem + ".png"
        else:
            image_filename = stem + ".png"

        image_path = self.image_dir / image_filename

        # Check if image file exists
        if image_path.exists():
            image = Image.open(image_path).convert("RGB")
            return image

        # Check if there's a directory with the same name (multi-page documents)
        dir_name = image_filename.replace(".png", "")
        dir_path = self.image_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            # Use first page for Stage 1 MVP
            first_page = dir_path / "1.png"
            if first_page.exists():
                logger.debug(f"Using first page from multi-page document: {first_page}")
                image = Image.open(first_page).convert("RGB")
                return image

        raise FileNotFoundError(
            f"Image not found: {image_path}. "
            f"Also checked multi-page directory: {dir_path}"
        )

    def _create_token_type_ids(self, seq_len: int, _question: str) -> torch.Tensor:
        """Create token type IDs (0 for all tokens).

        Stage 1: Simplified - use all 0s (LayoutLMv3 default type_vocab_size=1)
        Stage 2: Will properly implement type IDs if needed

        Args:
            seq_len: Sequence length
            _question: Question text (unused in Stage 1, reserved for Stage 2)

        Returns:
            Token type IDs tensor [seq_len]
        """
        # Stage 1 MVP: LayoutLMv3-base has type_vocab_size=1, only accepts 0
        # Use all zeros for compatibility
        token_type_ids = torch.zeros(seq_len, dtype=torch.long)

        return token_type_ids


def collate_fn_filter_none(batch):
    """Collate function that filters out None samples.

    Args:
        batch: List of samples, some may be None

    Returns:
        Collated batch with None samples filtered out
    """
    # Filter out None samples
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        # Return None if all samples failed
        return None

    # Use default collate for valid samples
    return torch.utils.data.dataloader.default_collate(batch)


def create_preprocessed_dataloader(
    json_dir: Path,
    image_dir: Path,
    output_path: Path,
    processor_name: str = "docling",
    batch_size: int = 8,
    max_length: int = DEFAULT_MAX_LENGTH,
    num_objects: int = DEFAULT_NUM_OBJECTS,
    require_all_verifiers: bool = True,
) -> DataLoader:
    """Create and save a preprocessed DataLoader from JSON QA pairs.

    This is the main entry point for preprocessing.

    Args:
        json_dir: Directory containing processed JSON files
        image_dir: Directory containing document images
        output_path: Path to save the preprocessed DataLoader pickle
        processor_name: Processor name to filter JSON files (e.g., "docling")
        batch_size: Batch size for DataLoader
        max_length: Maximum sequence length
        num_objects: Maximum number of objects per document
        require_all_verifiers: Whether all verifiers must say "Yes"

    Returns:
        Created DataLoader (also saved to output_path)
    """
    from transformers import AutoProcessor

    from docs2synth.retriever.dataset import load_verified_qa_pairs

    # Validate and set default batch_size if None
    if batch_size is None:
        batch_size = 8
        logger.warning(f"batch_size was None, using default: {batch_size}")
    elif batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got: {batch_size}")

    logger.info("=" * 70)
    logger.info("Starting preprocessing: JSON → DataLoader")
    logger.info("=" * 70)

    # 1. Load verified QA pairs
    logger.info(f"\n[1/5] Loading verified QA pairs from {json_dir}")
    qa_pairs = load_verified_qa_pairs(
        json_dir,
        processor_name=processor_name,
        require_all_verifiers=require_all_verifiers,
    )

    if not qa_pairs:
        raise ValueError(
            f"No verified QA pairs found in {json_dir}.\n"
            "Please ensure JSON files contain QA pairs with verification results."
        )

    logger.info(f"✓ Loaded {len(qa_pairs)} verified QA pairs")

    # 2. Create processor
    logger.info("\n[2/5] Loading LayoutLMv3 processor")
    processor = AutoProcessor.from_pretrained(
        "microsoft/layoutlmv3-base", apply_ocr=False
    )
    logger.info("✓ Processor loaded")

    # 3. Create dataset
    logger.info("\n[3/5] Creating PreprocessedQADataset")
    dataset = PreprocessedQADataset(
        qa_pairs=qa_pairs,
        processor=processor,
        image_dir=image_dir,
        max_length=max_length,
        num_objects=num_objects,
    )
    logger.info("✓ Dataset created")

    # 4. Create dataloader configuration
    logger.info(f"\n[4/5] Preparing dataset configuration (batch_size={batch_size})")
    dataloader_config = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": 0,  # Use 0 for debugging, can increase later
        "collate_fn": collate_fn_filter_none,  # Filter None samples
    }
    logger.info("✓ Configuration prepared")

    # 5. Save to pickle
    logger.info(f"\n[5/5] Saving dataset configuration to {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(dataloader_config, f)

    logger.info("✓ Dataset configuration saved successfully!")
    logger.info("\n" + "=" * 70)
    logger.info("Preprocessing complete!")
    logger.info("=" * 70)
    logger.info(f"Output: {output_path}")
    logger.info(f"QA pairs: {len(qa_pairs)}")
    logger.info(f"Batch size: {batch_size}")
    # Only calculate estimated batches if batch_size is valid
    if batch_size is not None and batch_size > 0:
        logger.info(
            f"Estimated batches per epoch: {(len(qa_pairs) + batch_size - 1) // batch_size}"
        )
    else:
        logger.warning(
            "Batch size is None or invalid, cannot estimate batches per epoch"
        )
    logger.info("\nUse with:")
    logger.info(f"  docs2synth retriever train --data-path {output_path}")
    logger.info("=" * 70)

    # Return a fresh DataLoader for immediate use if needed
    return DataLoader(**dataloader_config)


def test_preprocessing(json_dir: Path, image_dir: Path, num_samples: int = 3):
    """Test preprocessing on a few samples without saving.

    Useful for debugging and validation.

    Args:
        json_dir: Directory containing JSON files
        image_dir: Directory containing images
        num_samples: Number of samples to test
    """
    from transformers import AutoProcessor

    from docs2synth.retriever.dataset import load_verified_qa_pairs

    logger.info("Testing preprocessing on sample data...")

    # Load QA pairs
    qa_pairs = load_verified_qa_pairs(json_dir, processor_name="docling")
    if not qa_pairs:
        logger.error("No QA pairs found!")
        return

    logger.info(f"Testing on {min(num_samples, len(qa_pairs))} samples")

    # Create processor and dataset
    processor = AutoProcessor.from_pretrained(
        "microsoft/layoutlmv3-base", apply_ocr=False
    )
    dataset = PreprocessedQADataset(qa_pairs[:num_samples], processor, image_dir)

    # Test each sample
    for i in range(min(num_samples, len(dataset))):
        logger.info(f"\nSample {i+1}:")
        try:
            item = dataset[i]
            logger.info(f"  Question: {qa_pairs[i]['question']}")
            logger.info(f"  Answer: {qa_pairs[i]['answer']}")
            logger.info("  Tensor shapes:")
            for key, value in item.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"    {key}: {value.shape}")
            logger.info("  ✓ Successfully preprocessed")
        except Exception as e:
            logger.error(f"  ✗ Error: {e}")

    logger.info("\nTest complete!")
