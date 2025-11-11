"""State management for annotation tool using Streamlit session state."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st

from docs2synth.annotation.data_manager import AnnotationDataManager


def initialize_state(data_dir: Optional[str] = None):
    """Initialize session state variables.

    Args:
        data_dir: Directory containing JSON files
    """
    # Data manager
    if "data_manager" not in st.session_state:
        if data_dir:
            st.session_state.data_manager = AnnotationDataManager(Path(data_dir))
        else:
            st.session_state.data_manager = None

    # Navigation state
    if "current_file_idx" not in st.session_state:
        st.session_state.current_file_idx = 0

    if "current_obj_id" not in st.session_state:
        st.session_state.current_obj_id = None

    if "current_qa_idx" not in st.session_state:
        st.session_state.current_qa_idx = 0

    # UI state
    if "show_cropped" not in st.session_state:
        st.session_state.show_cropped = False

    if "filter_unannotated" not in st.session_state:
        st.session_state.filter_unannotated = False

    if "annotator_name" not in st.session_state:
        st.session_state.annotator_name = "anonymous"

    # Temp state for current annotation
    if "temp_explanation" not in st.session_state:
        st.session_state.temp_explanation = ""


def load_data_directory(
    data_dir: str,
    image_dirs: Optional[list] = None,
    processor_filter: Optional[str] = None,
):
    """Load data from directory.

    Args:
        data_dir: Directory containing JSON files
        image_dirs: Optional list of image directories
        processor_filter: Only load JSON files from this processor (e.g., 'easyocr', 'paddleocr')
    """
    st.session_state.data_manager = AnnotationDataManager(
        Path(data_dir),
        image_dirs=[Path(d) for d in image_dirs] if image_dirs else None,
        processor_filter=processor_filter,
    )
    num_files = st.session_state.data_manager.load_files()

    if num_files > 0:
        st.session_state.data_manager.load_document(0)
        st.session_state.current_file_idx = 0
        st.session_state.current_qa_idx = 0
        return True
    return False


def get_current_qa():
    """Get current QA pair and associated data.

    Returns:
        Tuple of (obj_id, qa_idx, qa_pair, obj, image) or None if not available
    """
    if not st.session_state.data_manager:
        return None

    doc = st.session_state.data_manager.current_document
    if not doc:
        return None

    dm = st.session_state.data_manager

    # Initialize current_obj_id if needed
    if st.session_state.current_obj_id is None:
        obj_ids = list(doc.objects.keys())
        if obj_ids:
            st.session_state.current_obj_id = obj_ids[0]
        else:
            # No objects - return placeholder to show image only
            st.session_state.current_obj_id = -1
            # Use first page for PDFs, or the single image
            image = dm.current_image
            return -1, 0, None, None, image

    # Handle case where there are no objects (obj_id = -1)
    if st.session_state.current_obj_id == -1:
        image = dm.current_image
        return -1, 0, None, None, image

    # Get current object
    obj_id = st.session_state.current_obj_id
    obj = doc.objects.get(obj_id)
    if not obj:
        return None

    # Get the appropriate image for this object
    # For PDFs, use the page-specific image; for single images, use the current_image
    if dm.is_pdf and dm.current_page_images:
        # Get page number from object (default to 0 if not set)
        obj_page = obj.page if obj.page is not None else 0
        image = dm.current_page_images.get(obj_page)
        if image is None:
            # Fallback to first page if object's page not found
            image = (
                next(iter(dm.current_page_images.values()))
                if dm.current_page_images
                else dm.current_image
            )
    else:
        # Single image document
        image = dm.current_image

    # Get QA pair (or None if no QA)
    qa_pair = None
    if obj.qa and len(obj.qa) > st.session_state.current_qa_idx:
        qa_pair = obj.qa[st.session_state.current_qa_idx]

    return obj_id, st.session_state.current_qa_idx, qa_pair, obj, image


def next_qa():
    """Move to next QA pair within current object."""
    if not st.session_state.data_manager:
        return

    doc = st.session_state.data_manager.current_document
    if not doc or st.session_state.current_obj_id is None:
        return

    obj = doc.objects.get(st.session_state.current_obj_id)
    if not obj or not obj.qa:
        return

    # Move to next QA in current object
    if st.session_state.current_qa_idx < len(obj.qa) - 1:
        st.session_state.current_qa_idx += 1


def prev_qa():
    """Move to previous QA pair within current object."""
    if not st.session_state.data_manager:
        return

    if st.session_state.current_qa_idx > 0:
        st.session_state.current_qa_idx -= 1


def next_object():
    """Move to next object in current document."""
    if not st.session_state.data_manager:
        return

    doc = st.session_state.data_manager.current_document
    if not doc:
        return

    obj_ids = list(doc.objects.keys())
    if not obj_ids:
        return

    # Find current object index
    if st.session_state.current_obj_id in obj_ids:
        current_idx = obj_ids.index(st.session_state.current_obj_id)
        if current_idx < len(obj_ids) - 1:
            st.session_state.current_obj_id = obj_ids[current_idx + 1]
            st.session_state.current_qa_idx = 0


def prev_object():
    """Move to previous object in current document."""
    if not st.session_state.data_manager:
        return

    doc = st.session_state.data_manager.current_document
    if not doc:
        return

    obj_ids = list(doc.objects.keys())
    if not obj_ids:
        return

    # Find current object index
    if st.session_state.current_obj_id in obj_ids:
        current_idx = obj_ids.index(st.session_state.current_obj_id)
        if current_idx > 0:
            st.session_state.current_obj_id = obj_ids[current_idx - 1]
            st.session_state.current_qa_idx = 0


def next_document():
    """Move to next document."""
    if not st.session_state.data_manager:
        return

    num_files = len(st.session_state.data_manager.json_files)
    if st.session_state.current_file_idx < num_files - 1:
        st.session_state.current_file_idx += 1
        st.session_state.data_manager.load_document(st.session_state.current_file_idx)
        st.session_state.current_obj_id = None  # Reset to first object
        st.session_state.current_qa_idx = 0


def prev_document():
    """Move to previous document."""
    if not st.session_state.data_manager:
        return

    if st.session_state.current_file_idx > 0:
        st.session_state.current_file_idx -= 1
        st.session_state.data_manager.load_document(st.session_state.current_file_idx)
        st.session_state.current_obj_id = None  # Reset to first object
        st.session_state.current_qa_idx = 0


def save_annotation(response: str, explanation: str = ""):
    """Save annotation for current QA pair.

    Args:
        response: "Yes" or "No"
        explanation: Optional explanation
    """
    if not st.session_state.data_manager:
        return False

    current = get_current_qa()
    if not current:
        return False

    obj_id, qa_idx, qa_pair, obj, image = current

    success = st.session_state.data_manager.save_annotation(
        obj_id=obj_id,
        qa_idx=qa_idx,
        response=response,
        explanation=explanation,
        annotator=st.session_state.annotator_name,
    )

    if success:
        # Clear explanation
        st.session_state.temp_explanation = ""

    return success


def get_qa_list_filtered():
    """Get filtered QA list based on current filters.

    Returns:
        List of (list_idx, obj_id, qa_idx, qa_pair) tuples
    """
    if not st.session_state.data_manager:
        return []

    qa_list = st.session_state.data_manager.get_qa_list()
    filtered = []

    for list_idx, (obj_id, qa_idx, qa_pair) in enumerate(qa_list):
        # Apply filters
        if st.session_state.filter_unannotated:
            # Only show unannotated QA pairs
            if qa_pair.verification and "human" in qa_pair.verification:
                continue

        filtered.append((list_idx, obj_id, qa_idx, qa_pair))

    return filtered


def jump_to_qa(list_idx: int):
    """Jump to specific QA pair by list index.

    Args:
        list_idx: Index in the QA list
    """
    st.session_state.current_qa_idx = list_idx


def get_global_qa_position() -> tuple[int, int]:
    """Return (current_index, total) for QA across the document list.

    Maps the current (obj_id, qa_idx) to its index within the flattened QA list
    provided by the data manager.
    """
    if not st.session_state.data_manager:
        return 0, 0

    current = get_current_qa()
    if not current:
        return 0, 0

    cur_obj_id, cur_qa_idx, _qa_pair, _obj, _image = current

    qa_list = st.session_state.data_manager.get_qa_list()
    total = len(qa_list)
    current_index = 0

    for idx, (obj_id, qa_idx, _pair) in enumerate(qa_list):
        if obj_id == cur_obj_id and qa_idx == cur_qa_idx:
            current_index = idx
            break

    return current_index, total
