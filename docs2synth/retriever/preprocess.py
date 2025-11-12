"""Preprocessing utilities for converting JSON QA pairs to training tensors.

This module provides functions to preprocess verified QA pairs from JSON files
into the tensor format required by the training functions.

Stage 1 (MVP): Implements basic fields with simplified versions for complex features.
Stage 2: Will implement full feature extraction (visual features, BERT embeddings, etc.)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List

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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a preprocessed example.

        Returns:
            Dictionary with all required tensor fields for training
        """
        qa_pair = self.qa_pairs[idx]

        try:
            return self._preprocess(qa_pair)
        except Exception as e:
            logger.error(f"Error preprocessing QA pair {idx}: {e}")
            logger.error(f"QA pair: {qa_pair.get('question', 'N/A')}")
            raise

    def _preprocess(self, qa: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Preprocess a single QA pair into training format.

        Stage 1 (MVP) implementation with simplified features.

        Args:
            qa: QA pair dictionary with keys:
                - question, answer, object_text, context
                - bbox, json_file, etc.

        Returns:
            Dictionary with all required tensor fields
        """
        # 1. Load image
        image = self._load_image(qa)

        # 2. Prepare text
        question = qa["question"]

        # 3. Stage 1 MVP: Use detected tokenizer (cached at initialization)
        text_encoding = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )

        # Process image separately
        image_processor = self.processor.image_processor
        image_encoding = image_processor(
            images=image,
            return_tensors="pt",
        )

        # Extract basic fields
        result = {
            "input_ids": text_encoding["input_ids"].squeeze(0),
            "attention_mask": text_encoding["attention_mask"].squeeze(0),
            "pixel_values": image_encoding["pixel_values"].squeeze(0),
        }

        seq_len = result["input_ids"].size(0)

        # Create dummy bounding boxes for tokens (Stage 1: all at [0, 0, 0, 0])
        # Stage 2 will properly map tokens to OCR bboxes
        result["bbox"] = torch.zeros(seq_len, 4, dtype=torch.long)

        # 4. Add simplified fields (Stage 1 - MVP)
        result.update(
            {
                # Token type IDs: 0 for question, 1 for context (simplified)
                "token_type_ids": self._create_token_type_ids(seq_len, question),
                # Token to object mapping (simplified: all tokens map to object 0)
                "token_objt_ids": torch.zeros(seq_len, dtype=torch.long),
                # Visual features (Stage 1: use zero vectors, Stage 2: extract from vision model)
                "visual_feat": torch.zeros(
                    self.num_objects, DEFAULT_HIDDEN_SIZE, dtype=torch.float
                ),
                # BERT CLS embedding (Stage 1: zero vector, Stage 2: extract from BERT)
                "bert_cls": torch.zeros(DEFAULT_HIDDEN_SIZE, dtype=torch.float),
                # Positional encoding (Stage 1: simple, Stage 2: based on bbox positions)
                "positional_encoding": torch.zeros(
                    self.num_objects, DEFAULT_POSITIONAL_ENCODING_DIM, dtype=torch.float
                ),
                # Normalized bounding boxes (Stage 1: zeros, Stage 2: from JSON objects)
                "norm_bbox": torch.zeros(self.num_objects, 4, dtype=torch.float),
                # Object mask (Stage 1: all valid, Stage 2: based on actual objects)
                "object_mask": torch.ones(self.num_objects, dtype=torch.float),
                # Entity target: which object contains the answer (Stage 1: object 0)
                "target": self._create_entity_target(),
                # Answer span positions (Stage 1: find in tokens, Stage 2: improve accuracy)
                **self._find_answer_positions(qa, text_encoding),
            }
        )

        return result

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

    def _create_entity_target(self) -> torch.Tensor:
        """Create entity target (which object contains the answer).

        Stage 1: Simplified - assume object 0 contains answer
        Stage 2: Will find actual object with answer

        Returns:
            One-hot tensor [num_objects]
        """
        target = torch.zeros(self.num_objects, dtype=torch.float)
        target[0] = 1.0  # Assume object 0 contains answer
        return target

    def _find_answer_positions(
        self, qa: Dict[str, Any], encoding: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Find answer start and end positions in tokenized sequence.

        Stage 1: Simple substring search
        Stage 2: Will improve with better alignment

        Args:
            qa: QA pair with answer text
            encoding: Processor output with input_ids

        Returns:
            Dictionary with start_id and end_id tensors
        """
        answer = qa["answer"]
        input_ids = encoding["input_ids"].squeeze(0)

        # Tokenize the answer using cached tokenizer
        answer_encoding = self.tokenizer(
            answer, add_special_tokens=False, return_tensors="pt"
        )
        answer_ids = answer_encoding["input_ids"].squeeze(0)

        # Find answer in input_ids (simple sliding window)
        start_pos, end_pos = self._find_sublist(input_ids, answer_ids)

        return {
            "start_id": torch.tensor(start_pos, dtype=torch.long),
            "end_id": torch.tensor(end_pos, dtype=torch.long),
        }

    def _find_sublist(
        self, main_list: torch.Tensor, sub_list: torch.Tensor
    ) -> tuple[int, int]:
        """Find sublist in main list (for answer span finding).

        Args:
            main_list: Input token IDs
            sub_list: Answer token IDs

        Returns:
            Tuple of (start_pos, end_pos)
        """
        if len(sub_list) == 0:
            return 0, 0

        main = main_list.tolist()
        sub = sub_list.tolist()

        # Sliding window search
        for i in range(len(main) - len(sub) + 1):
            if main[i : i + len(sub)] == sub:
                return i, i + len(sub) - 1

        # If not found, return middle of sequence as fallback
        mid = len(main) // 2
        return mid, min(mid + 5, len(main) - 1)


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

    # 4. Create dataloader
    logger.info(f"\n[4/5] Creating DataLoader (batch_size={batch_size})")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for debugging, can increase later
        collate_fn=None,  # Use default collate
    )
    logger.info("✓ DataLoader created")

    # 5. Save to pickle
    logger.info(f"\n[5/5] Saving DataLoader to {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(dataloader, f)

    logger.info("✓ DataLoader saved successfully!")
    logger.info("\n" + "=" * 70)
    logger.info("Preprocessing complete!")
    logger.info("=" * 70)
    logger.info(f"Output: {output_path}")
    logger.info(f"QA pairs: {len(qa_pairs)}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Batches: {len(dataloader)}")
    logger.info("\nUse with:")
    logger.info(f"  docs2synth retriever train --data-path {output_path}")
    logger.info("=" * 70)

    return dataloader


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
