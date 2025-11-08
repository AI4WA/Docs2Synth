"""Schema models for document processing outputs.

These models describe the structured output returned by the document processing
pipeline, including recognized objects (e.g., text boxes), their labels, and
associated metadata about both the processing run and the input document.

The models are implemented using standard library types to avoid introducing
runtime dependencies. They provide explicit serialization helpers for
round-tripping to and from dictionaries/JSON.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

# A bounding box is represented as (x1, y1, x2, y2) with float coordinates.
BBox = Tuple[float, float, float, float]

# A polygon is represented as a list of (x, y) coordinate pairs.
# For quadrilaterals (common in OCR), this will have 4 points: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
Polygon = List[Tuple[float, float]]


class LabelType(str, Enum):
    """Semantic granularity label of an extracted object.

    Describes the semantic unit represented by a `DocumentObject`.

    Notes
    -----
    The set below covers common granularities. Pipelines can downcast unknown or
    custom values to `OTHER` when loading from dicts.
    """

    TEXT = "text"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    DOCUMENT = "document"
    OTHER = "other"


@dataclass
class QAPair:
    """A question-answer pair associated with a document object.

    Attributes
    ----------
    question : str
        The generated question.
    answer : Optional[str]
        The answer to the question. Can be None or the text field from the object.
    strategy : Optional[str]
        The QA generation strategy used (e.g., "semantic", "layout_aware").
    verification : Optional[Dict[str, Dict[str, Any]]]
        Verification results from different verifiers (e.g., {"meaningful": {...}, "correctness": {...}}).
    extra : Dict[str, Any]
        Free-form additional attributes (e.g., provider, model).
    """

    question: str
    answer: Optional[str] = None
    strategy: Optional[str] = None
    verification: Optional[Dict[str, Dict[str, Any]]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "question": self.question,
        }
        if self.answer is not None:
            data["answer"] = self.answer
        if self.strategy is not None:
            data["strategy"] = self.strategy
        if self.verification is not None:
            data["verification"] = self.verification
        if self.extra:
            data.update(self.extra)
        return data

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "QAPair":
        known_keys = {"question", "answer", "strategy", "verification"}
        extra = {k: v for k, v in data.items() if k not in known_keys}
        return QAPair(
            question=str(data.get("question", "")),
            answer=data.get("answer"),
            strategy=data.get("strategy"),
            verification=data.get("verification"),
            extra=extra,
        )


@dataclass
class DocumentObject:
    """A single extracted object from a document (e.g., an OCR text box).

    Attributes
    ----------
    object_id : int
        Stable identifier of the object within the result.
    text : str
        Text content associated with the object.
    bbox : Tuple[float, float, float, float]
        Bounding box as `(x1, y1, x2, y2)` in document/page coordinates.
        This is the axis-aligned bounding box (for backward compatibility).
    polygon : Optional[List[Tuple[float, float]]]
        Optional polygon coordinates as list of (x, y) points.
        For OCR detections, this is typically a quadrilateral with 4 points.
        If provided, this is more accurate than bbox for rotated text.
    label : LabelType
        Semantic label describing the object's granularity.
    page : Optional[int]
        Optional page index (0-based) from which the object was extracted.
    score : Optional[float]
        Optional confidence/probability score emitted by the extractor.
    qa : List[QAPair]
        List of question-answer pairs generated for this object.
    extra : Dict[str, Any]
        Free-form additional attributes emitted by the pipeline.
    """

    object_id: int
    text: str
    bbox: BBox
    label: LabelType = LabelType.TEXT
    polygon: Optional[Polygon] = None
    page: Optional[int] = None
    score: Optional[float] = None
    qa: List[QAPair] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "text": self.text,
            "bbox": list(self.bbox),
            "label": str(
                self.label.value if isinstance(self.label, LabelType) else self.label
            ),
        }
        if self.polygon is not None:
            data["polygon"] = [list(point) for point in self.polygon]
        if self.page is not None:
            data["page"] = self.page
        if self.score is not None:
            data["score"] = self.score
        if self.qa:
            data["qa"] = [qa_pair.to_dict() for qa_pair in self.qa]
        if self.extra:
            data.update(self.extra)
        return data

    @staticmethod
    def from_dict(object_id: int, data: Mapping[str, Any]) -> "DocumentObject":
        label_value = data.get("label", LabelType.TEXT)
        label = (
            LabelType(label_value)
            if label_value in LabelType._value2member_map_
            else LabelType.OTHER
        )
        bbox_list = data.get("bbox", [0.0, 0.0, 0.0, 0.0])
        bbox: BBox = (
            float(bbox_list[0]),
            float(bbox_list[1]),
            float(bbox_list[2]),
            float(bbox_list[3]),
        )
        text_value = str(data.get("text", ""))

        # Parse polygon if present
        polygon: Optional[Polygon] = None
        if "polygon" in data and isinstance(data["polygon"], list):
            polygon = [(float(p[0]), float(p[1])) for p in data["polygon"]]

        # Parse QA pairs if present
        qa_list: List[QAPair] = []
        if "qa" in data and isinstance(data["qa"], list):
            qa_list = [QAPair.from_dict(qa_data) for qa_data in data["qa"]]

        # Extract known fields and keep the rest in extra
        known_keys = {"text", "bbox", "polygon", "label", "page", "score", "qa"}
        extra = {k: v for k, v in data.items() if k not in known_keys}

        return DocumentObject(
            object_id=object_id,
            text=text_value,
            bbox=bbox,
            polygon=polygon,
            label=label,
            page=(
                int(data["page"])
                if "page" in data and data["page"] is not None
                else None
            ),
            score=(
                float(data["score"])
                if "score" in data and data["score"] is not None
                else None
            ),
            qa=qa_list,
            extra=extra,
        )


@dataclass
class ProcessMetadata:
    """Metadata describing how the document was processed.

    Attributes
    ----------
    processor_name : Optional[str]
        Name/identifier of the processor component (e.g., "tesseract-ocr", "layoutlm").
    timestamp : str
        Start time of the processing run in ISO-8601 format (UTC, suffixed with "Z").
    latency : Optional[float]
        Total processing latency in milliseconds.
    """

    # Name/identifier of the processor component (e.g., "tesseract-ocr", "layoutlm")
    processor_name: Optional[str] = None
    # this should be the start time of the processing run
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    latency: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "processor_name": self.processor_name,
            "timestamp": self.timestamp,
            "latency": self.latency,
        }
        # Drop None values
        return {k: v for k, v in data.items() if v is not None}


@dataclass
class DocumentMetadata:
    """Metadata about the given input document.

    Attributes
    ----------
    source : Optional[str]
        Location of the input document (e.g., local path or URL).
    filename : Optional[str]
        Basename of the document if available.
    document_id : Optional[str]
        Upstream identifier for the document.
    page_count : Optional[int]
        Number of pages, if known.
    size_bytes : Optional[int]
        File size in bytes, if known.
    mime_type : Optional[str]
        MIME type of the input (e.g., "image/png", "application/pdf").
    language : Optional[str]
        Primary language of the document (BCP-47 or ISO code), if known.
    width : Optional[int]
        For single-page images: pixel width of the document.
    height : Optional[int]
        For single-page images: pixel height of the document.
    extra : Dict[str, Any]
        Free-form additional metadata (custom fields, per-page details, etc.).
    """

    source: Optional[str] = None  # e.g., file path or URL
    filename: Optional[str] = None
    document_id: Optional[str] = None
    page_count: Optional[int] = None
    size_bytes: Optional[int] = None
    mime_type: Optional[str] = None
    language: Optional[str] = None
    width: Optional[int] = (
        None  # for single-page images; per-page widths should go to extra
    )
    height: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Remove None values; keep non-empty extra only
        cleaned: Dict[str, Any] = {
            k: v for k, v in data.items() if v is not None and (k != "extra" or bool(v))
        }
        if "extra" not in cleaned:
            cleaned["extra"] = {}
        return cleaned


@dataclass
class DocumentProcessResult:
    """Full result of a document processing run.

    Encapsulates recognized objects, convenience lists, overall context, reading
    order, and both processing and document metadata.

    Attributes
    ----------
    objects : Dict[int, DocumentObject]
        Mapping from object id to `DocumentObject`.
    object_list : List[DocumentObject]
        Flat list of objects (useful when ordering by reading order or index).
    bbox_list : List[Tuple[float, float, float, float]]
        List of bounding boxes corresponding to recognized objects.
    context : str
        Concatenated context string assembled from recognized objects.
    reading_order_ids : List[int]
        Object ids in the intended reading order.
    process_metadata : ProcessMetadata
        Metadata describing this processing run.
    document_metadata : DocumentMetadata
        Metadata about the source document.
    """

    objects: Dict[int, DocumentObject] = field(default_factory=dict)
    object_list: List[DocumentObject] = field(default_factory=list)
    bbox_list: List[BBox] = field(default_factory=list)
    context: str = ""
    reading_order_ids: List[int] = field(default_factory=list)
    process_metadata: ProcessMetadata = field(default_factory=ProcessMetadata)
    document_metadata: DocumentMetadata = field(default_factory=DocumentMetadata)

    def to_dict(self) -> Dict[str, Any]:
        # Serialize objects both as a mapping and as a list to match expected shape
        objects_dict: Dict[int, Dict[str, Any]] = {
            obj_id: obj.to_dict() for obj_id, obj in self.objects.items()
        }
        object_list_serialized: List[Dict[str, Any]] = [
            obj.to_dict() for obj in self.object_list
        ]
        bbox_list_serialized: List[List[float]] = [list(b) for b in self.bbox_list]

        return {
            "objects": objects_dict,
            "object_list": object_list_serialized,
            "bbox_list": bbox_list_serialized,
            "context": self.context,
            "reading_order_ids": list(self.reading_order_ids),
            "process_metadata": self.process_metadata.to_dict(),
            "document_metadata": self.document_metadata.to_dict(),
        }

    def to_json(self, *, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "DocumentProcessResult":
        # Objects
        objects_raw: Mapping[str, Any] = data.get("objects", {})
        objects: Dict[int, DocumentObject] = {}
        for key, obj_data in objects_raw.items():
            try:
                obj_id = int(key)
            except Exception:
                continue
            objects[obj_id] = DocumentObject.from_dict(obj_id, obj_data)

        # Object list (pure: enumerate sequentially without cross-referencing ids)
        object_list_data: Iterable[Mapping[str, Any]] = data.get("object_list", [])
        object_list: List[DocumentObject] = [
            DocumentObject.from_dict(i, item) for i, item in enumerate(object_list_data)
        ]

        # BBox list
        bbox_list_raw = data.get("bbox_list", [])
        bbox_list: List[BBox] = [
            (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            for b in bbox_list_raw
            if isinstance(b, (list, tuple)) and len(b) == 4
        ]

        # Metadata
        process_metadata_raw = data.get("process_metadata", {})
        document_metadata_raw = data.get("document_metadata", {})

        process_metadata = ProcessMetadata(
            processor_name=process_metadata_raw.get("processor_name"),
            timestamp=process_metadata_raw.get(
                "timestamp", datetime.utcnow().isoformat() + "Z"
            ),
            latency=process_metadata_raw.get("latency"),
        )

        document_metadata = DocumentMetadata(
            source=document_metadata_raw.get("source"),
            filename=document_metadata_raw.get("filename"),
            document_id=document_metadata_raw.get("document_id"),
            page_count=document_metadata_raw.get("page_count"),
            size_bytes=document_metadata_raw.get("size_bytes"),
            mime_type=document_metadata_raw.get("mime_type"),
            language=document_metadata_raw.get("language"),
            width=document_metadata_raw.get("width"),
            height=document_metadata_raw.get("height"),
            extra=dict(document_metadata_raw.get("extra", {})),
        )

        return DocumentProcessResult(
            objects=objects,
            object_list=object_list,
            bbox_list=bbox_list,
            context=str(data.get("context", "")),
            reading_order_ids=list(data.get("reading_order_ids", [])),
            process_metadata=process_metadata,
            document_metadata=document_metadata,
        )


__all__ = [
    "BBox",
    "LabelType",
    "DocumentObject",
    "ProcessMetadata",
    "DocumentMetadata",
    "DocumentProcessResult",
]
