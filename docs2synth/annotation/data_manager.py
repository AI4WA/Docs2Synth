"""Data manager for annotation tool.

Handles loading and saving of JSON files and images.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image as PILImage

from docs2synth.preprocess.schema import DocumentProcessResult, QAPair
from docs2synth.qa.verify_batch import find_image_for_json

logger = logging.getLogger(__name__)


class AnnotationDataManager:
    """Manages loading and saving annotation data."""

    def __init__(
        self,
        data_dir: Path,
        image_dirs: Optional[List[Path]] = None,
        processor_filter: Optional[str] = None,
    ):
        """Initialize data manager.

        Args:
            data_dir: Directory containing JSON files
            image_dirs: Directories to search for images (default: same as data_dir)
            processor_filter: Only load JSON files from this processor (e.g., 'paddleocr', 'easyocr')
        """
        self.data_dir = Path(data_dir)
        self.image_dirs = image_dirs or [self.data_dir]
        self.processor_filter = processor_filter

        self.json_files: List[Path] = []
        self.current_file_idx = 0
        self.current_document: Optional[DocumentProcessResult] = None
        self.current_image: Optional[PILImage.Image] = None
        self.current_image_path: Optional[Path] = None

    def load_files(self) -> int:
        """Load all JSON files from data directory.

        If processor_filter is set, only loads files matching the pattern:
        *_{processor_filter}.json (e.g., *_easyocr.json, *_paddleocr.json)

        Returns:
            Number of files loaded
        """
        if self.processor_filter:
            # Load only files matching the processor pattern
            pattern = f"*_{self.processor_filter}.json"
            self.json_files = sorted(self.data_dir.glob(pattern))
            logger.info(
                f"Found {len(self.json_files)} {self.processor_filter} JSON files in {self.data_dir}"
            )
        else:
            # Load all JSON files
            self.json_files = sorted(self.data_dir.glob("*.json"))
            logger.info(f"Found {len(self.json_files)} JSON files in {self.data_dir}")

        return len(self.json_files)

    def load_document(self, file_idx: int) -> bool:
        """Load a specific document by index.

        Args:
            file_idx: Index of the file to load

        Returns:
            True if successful, False otherwise
        """
        if not self.json_files or file_idx < 0 or file_idx >= len(self.json_files):
            logger.error(f"Invalid file index: {file_idx}")
            return False

        json_path = self.json_files[file_idx]
        self.current_file_idx = file_idx

        try:
            # Load JSON
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.current_document = DocumentProcessResult.from_dict(data)

            # Load corresponding image
            image_path = find_image_for_json(json_path, self.image_dirs)
            if image_path:
                try:
                    self.current_image = PILImage.open(image_path)
                    # Verify image is loadable by trying to load the data
                    self.current_image.load()
                    self.current_image_path = image_path
                    logger.info(f"Loaded document: {json_path.name}")
                    logger.info(f"Loaded image: {image_path.name}")
                    return True
                except (OSError, SyntaxError) as e:
                    logger.error(f"Failed to load image {image_path}: {e}")
                    logger.warning("Creating placeholder for corrupted image")
                    # Create a placeholder image
                    self.current_image = PILImage.new(
                        "RGB", (800, 600), color="lightgray"
                    )
                    from PIL import ImageDraw

                    draw = ImageDraw.Draw(self.current_image)
                    draw.text(
                        (50, 300),
                        f"Image Load Error\n{image_path.name}\n{str(e)[:50]}",
                        fill="red",
                    )
                    self.current_image_path = image_path
                    return True
            else:
                logger.warning(f"Image not found for {json_path.name}")
                return False

        except Exception as e:
            logger.error(f"Failed to load document {json_path}: {e}")
            return False

    def get_qa_list(self) -> List[Tuple[int, int, Optional[QAPair]]]:
        """Get list of all QA pairs in current document, or objects without QA.

        Returns:
            List of (object_id, qa_idx, qa_pair) tuples
            - If object has QA pairs: returns one tuple per QA pair
            - If object has no QA pairs: returns one tuple with qa_pair=None
            - If document has no objects: returns one tuple to show the image
        """
        if not self.current_document:
            return []

        qa_list = []
        for obj_id, obj in self.current_document.objects.items():
            if obj.qa and len(obj.qa) > 0:
                # Object has QA pairs
                for qa_idx, qa_pair in enumerate(obj.qa):
                    qa_list.append((obj_id, qa_idx, qa_pair))
            else:
                # Object has no QA pairs, still show it
                qa_list.append((obj_id, 0, None))

        # If no objects at all, add a placeholder entry so we can still show the image
        if not qa_list:
            qa_list.append((-1, 0, None))  # -1 indicates no object

        return qa_list

    def save_annotation(
        self,
        obj_id: int,
        qa_idx: int,
        response: str,
        explanation: str = "",
        annotator: str = "anonymous",
    ) -> bool:
        """Save human annotation to current document.

        Args:
            obj_id: Object ID
            qa_idx: QA pair index within the object
            response: "Yes" or "No"
            explanation: Optional explanation
            annotator: Annotator username

        Returns:
            True if successful, False otherwise
        """
        if not self.current_document:
            return False

        try:
            obj = self.current_document.objects.get(obj_id)
            if not obj or not obj.qa or qa_idx >= len(obj.qa):
                logger.error(f"Invalid object ID or QA index: {obj_id}, {qa_idx}")
                return False

            qa_pair = obj.qa[qa_idx]

            # Initialize verification dict if not exists
            if qa_pair.verification is None:
                qa_pair.verification = {}

            # Add human annotation
            qa_pair.verification["human"] = {
                "verifier_type": "human",
                "response": response,
                "explanation": explanation,
                "annotator": annotator,
                "timestamp": datetime.now().isoformat(),
            }

            # Save to JSON file
            json_path = self.json_files[self.current_file_idx]
            output_data = self.current_document.to_dict()
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            logger.info(
                f"Saved annotation for {json_path.name} obj={obj_id} qa={qa_idx}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to save annotation: {e}")
            return False

    def get_stats(self) -> dict:
        """Get annotation statistics.

        Returns:
            Dictionary with statistics including objects without QA
        """
        if not self.current_document:
            return {}

        total_objects = len(self.current_document.objects)
        objects_with_qa = 0
        objects_without_qa = 0
        total_qa = 0
        annotated = 0
        yes_count = 0
        no_count = 0

        for obj in self.current_document.objects.values():
            if obj.qa and len(obj.qa) > 0:
                objects_with_qa += 1
                for qa_pair in obj.qa:
                    total_qa += 1
                    if qa_pair.verification and "human" in qa_pair.verification:
                        annotated += 1
                        response = qa_pair.verification["human"].get("response", "")
                        if response == "Yes":
                            yes_count += 1
                        elif response == "No":
                            no_count += 1
            else:
                objects_without_qa += 1

        return {
            "total_objects": total_objects,
            "objects_with_qa": objects_with_qa,
            "objects_without_qa": objects_without_qa,
            "total_qa": total_qa,
            "annotated": annotated,
            "not_annotated": total_qa - annotated,
            "yes_count": yes_count,
            "no_count": no_count,
            "progress": (annotated / total_qa * 100) if total_qa > 0 else 0,
        }
