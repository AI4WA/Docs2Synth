"""Streamlit app for human annotation of QA pairs.

Usage:
    streamlit run docs2synth/annotation/streamlit_app.py
"""

from __future__ import annotations

import os

import streamlit as st

from docs2synth.annotation import annotation_state as state
from docs2synth.annotation.image_renderer import (
    crop_object_region,
    render_image_with_bbox_pil,
)


def render_sidebar():
    """Render sidebar with configuration and navigation."""
    with st.sidebar:
        st.title("📝 QA Annotation Tool")

        # Data loading section
        st.header("🗂️ Data")

        # Get defaults from config.yml or environment variables
        from docs2synth.utils.config import Config

        try:
            config = Config.from_yaml("config.yml")
            default_data_dir = (
                os.getenv("DOCS2SYNTH_DATA_DIR")
                or config.get("preprocess.output_dir")
                or "./data/processed/dev"
            )
            default_image_dir = (
                os.getenv("DOCS2SYNTH_IMAGE_DIR")
                or config.get("preprocess.input_dir")
                or ""
            )
            default_processor = config.get("preprocess.processor", "paddleocr")
        except Exception:
            default_data_dir = os.getenv("DOCS2SYNTH_DATA_DIR", "./data/processed/dev")
            default_image_dir = os.getenv("DOCS2SYNTH_IMAGE_DIR", "")
            default_processor = "paddleocr"

        data_dir = st.text_input("Data Directory (JSON files)", value=default_data_dir)

        image_dir_input = st.text_input(
            "Image Directory (optional)",
            value=default_image_dir,
            placeholder="Leave empty to auto-detect from data directory",
        )

        # Processor filter
        processor_options = ["all", "paddleocr", "easyocr", "pdfplumber", "docling"]
        default_idx = (
            processor_options.index(default_processor)
            if default_processor in processor_options
            else 0
        )

        processor_filter = st.selectbox(
            "OCR Processor Filter",
            options=processor_options,
            index=default_idx,
            help="Only load JSON files from this processor (from config.yml)",
        )

        if st.button("📂 Load Data", width="stretch"):
            with st.spinner("Loading data..."):
                image_dirs = [image_dir_input] if image_dir_input else None
                # Convert "all" to None (load all files)
                proc_filter = None if processor_filter == "all" else processor_filter
                success = state.load_data_directory(data_dir, image_dirs, proc_filter)
                if success:
                    filter_msg = f" ({processor_filter} only)" if proc_filter else ""
                    st.success(
                        f"Loaded {len(st.session_state.data_manager.json_files)} files{filter_msg}!"
                    )
                    st.rerun()
                else:
                    st.error(f"No {processor_filter} JSON files found in directory!")

        st.divider()

        # Progress and stats
        if st.session_state.data_manager:
            st.header("📊 Progress")

            # Document progress
            num_files = len(st.session_state.data_manager.json_files)
            st.write(
                f"**Document:** {st.session_state.current_file_idx + 1} / {num_files}"
            )

            # QA progress
            qa_list = st.session_state.data_manager.get_qa_list()
            total_qa = len(qa_list)
            st.write(f"**QA Pair:** {st.session_state.current_qa_idx + 1} / {total_qa}")

            # Progress bar
            if total_qa > 0:
                progress = (st.session_state.current_qa_idx + 1) / total_qa
                st.progress(progress)

            # Annotation stats
            stats = st.session_state.data_manager.get_stats()

            # Show object statistics
            if stats.get("objects_without_qa", 0) > 0:
                st.warning(f"⚠️ {stats['objects_without_qa']} objects without QA")

            st.metric("Total Objects", stats.get("total_objects", 0))
            st.metric("Annotated QA", f"{stats['annotated']}/{stats['total_qa']}")

            if stats["annotated"] > 0:
                st.metric("Approval Rate", f"{stats['yes_count']}/{stats['annotated']}")

            st.divider()

            # Settings
            st.header("⚙️ Settings")

            st.session_state.annotator_name = st.text_input(
                "Annotator Name",
                value=st.session_state.annotator_name,
            )

            st.session_state.show_cropped = st.checkbox(
                "Show Cropped Region",
                value=st.session_state.show_cropped,
            )

            st.session_state.filter_unannotated = st.checkbox(
                "Filter Unannotated Only",
                value=st.session_state.filter_unannotated,
            )


def render_verifier_results(qa_pair):
    """Render existing verifier results.

    Args:
        qa_pair: QAPair instance or None
    """
    if qa_pair is None or not qa_pair.verification:
        st.info("No automatic verifier results available.")
        return

    st.subheader("🤖 Automatic Verifiers")

    # Filter out human annotations for display
    auto_verifiers = {k: v for k, v in qa_pair.verification.items() if k != "human"}

    if not auto_verifiers:
        st.info("No automatic verifier results available.")
        return

    cols = st.columns(len(auto_verifiers))

    for idx, (v_type, v_result) in enumerate(auto_verifiers.items()):
        with cols[idx]:
            response = v_result.get("response", "Unknown")
            explanation = v_result.get("explanation", "")

            # Color based on response
            if response.lower() == "yes":
                color = "green"
                emoji = "✅"
            elif response.lower() == "no":
                color = "red"
                emoji = "❌"
            else:
                color = "gray"
                emoji = "❓"

            st.markdown(f"### {emoji} {v_type.title()}")
            st.markdown(f":{color}[**{response}**]")

            if explanation:
                with st.expander("Explanation"):
                    st.write(explanation)


def _render_qa_progress():
    """Render progress bar for current QA."""
    current = state.get_current_qa()
    if current:
        dm = st.session_state.data_manager
        qa_list = dm.get_qa_list()
        current_idx = st.session_state.current_qa_idx
        total_qa = len(qa_list)

        # Mini progress bar at top
        progress_pct = (current_idx + 1) / total_qa if total_qa > 0 else 0
        st.progress(progress_pct, text=f"QA {current_idx + 1} of {total_qa}")
        st.caption(
            f"📊 Progress: {current_idx + 1}/{total_qa} ({progress_pct*100:.1f}%)"
        )


def _render_qa_content(qa_pair, obj):
    """Render question, answer, and additional info."""
    # Question - large and prominent
    st.markdown("### 🔵 Question")
    st.info(f"**{qa_pair.question}**", icon="❓")

    # Answer - large and prominent
    st.markdown("### 🟢 Answer")
    st.success(f"**{qa_pair.answer}**", icon="💬")

    # Additional info - collapsed by default
    with st.expander("📋 Additional Info", expanded=False):
        # Show object text if available
        if obj.text:
            st.markdown("**📄 Full Object Text:**")
            st.text_area(
                "",
                value=obj.text,
                height=100,
                disabled=True,
                label_visibility="collapsed",
            )

        # Verifier results
        if qa_pair.verification:
            num_verifiers = len(
                [k for k in qa_pair.verification.keys() if k != "human"]
            )
            if num_verifiers > 0:
                st.markdown(f"**🤖 Verifier Results ({num_verifiers}):**")
                render_verifier_results(qa_pair)


def _render_annotation_status(qa_pair):
    """Render existing annotation status if present."""
    is_annotated = qa_pair.verification and "human" in qa_pair.verification

    if is_annotated:
        human_result = qa_pair.verification["human"]
        response = human_result["response"]
        emoji = "✅" if response.lower() == "yes" else "❌"

        # Big status banner
        st.markdown(f"### {emoji} Already Annotated: **{response}**")
        st.caption(
            f"👤 By {human_result.get('annotator', 'unknown')} at {human_result.get('timestamp', 'N/A')}"
        )

        if human_result.get("explanation"):
            with st.expander("💬 View Explanation"):
                st.write(human_result["explanation"])

        st.info("💡 You can re-annotate to update the response", icon="ℹ️")
        st.divider()


def _handle_annotation_button(response: str, auto_advance: bool):
    """Handle annotation button click."""
    explanation = st.session_state.temp_explanation
    if state.save_annotation(response, explanation):
        st.session_state.temp_explanation = ""  # Clear explanation
        if auto_advance:
            # Auto advance to next QA
            state.next_qa()
            if state.get_current_qa() is None:  # Reached end of object
                state.next_object()
        emoji = "✅" if response == "Yes" else "❌"
        status = "Approved!" if response == "Yes" else "Rejected!"
        st.success(f"{emoji} {status}")
        st.rerun()
    else:
        st.error("Failed to save")


def _render_annotation_buttons(auto_advance: bool):
    """Render annotation buttons and settings."""
    st.subheader("👤 Quick Annotate")

    # Settings in one row
    col1, col2 = st.columns(2)

    with col1:
        # Auto-advance setting
        if "auto_advance" not in st.session_state:
            st.session_state.auto_advance = True

        auto_advance = st.checkbox(
            "🚀 Auto-advance",
            value=st.session_state.auto_advance,
            key="auto_advance_toggle",
            help="Automatically move to next QA after saving",
        )
        st.session_state.auto_advance = auto_advance

    with col2:
        # Show annotation count
        stats = (
            st.session_state.data_manager.get_stats()
            if st.session_state.data_manager
            else {}
        )
        annotated = stats.get("annotated", 0)
        total = stats.get("total_qa", 0)
        st.metric(
            "✅ Annotated",
            f"{annotated}/{total}",
            delta=None if total == 0 else f"{annotated/total*100:.0f}%",
        )

    # Big buttons - most common action
    st.markdown("### Is this QA pair correct?")

    # Large approval buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "✅ CORRECT",
            width="stretch",
            type="primary",
            key="approve_btn",
            use_container_width=True,
        ):
            _handle_annotation_button("Yes", auto_advance)

    with col2:
        if st.button(
            "❌ WRONG",
            width="stretch",
            key="reject_btn",
            use_container_width=True,
        ):
            _handle_annotation_button("No", auto_advance)

    st.divider()

    # Optional explanation at bottom
    with st.expander("💬 Add explanation (optional)", expanded=False):
        explanation = st.text_area(
            "Why is this correct/wrong?",
            value=st.session_state.temp_explanation,
            height=80,
            placeholder="Optional: Explain your reasoning...",
            key="explanation_input",
        )
        st.session_state.temp_explanation = explanation

    # Keyboard shortcuts hint
    st.caption("💡 Tip: Click 'CORRECT' for good QA, 'WRONG' for bad QA")


def render_qa_panel():
    """Render QA pair display and annotation interface."""
    current = state.get_current_qa()

    if not current:
        st.warning("No QA pairs available. Please load data first.")
        return

    obj_id, qa_idx, qa_pair, obj, image = current

    # Check if this is an empty document (no objects)
    if obj_id == -1:
        st.info("📭 No text objects detected in this document")
        st.caption(
            "The OCR processor did not find any text in this image. You can still view the image in the left panel."
        )
        return

    # Check if this object has QA pairs
    if qa_pair is None:
        st.warning(f"⚠️ Object {obj_id} has no QA pairs yet")
        st.info(
            "This object was detected but no questions were generated. You can still view the object text and bounding box."
        )

        # Show object text
        if obj and obj.text:
            st.subheader("📄 Object Text")
            st.text_area("Text content", value=obj.text, height=150, disabled=True)

        # Show bbox info
        if obj and obj.bbox:
            st.caption(f"Bounding Box: {obj.bbox}")
            st.caption(f"Object ID: {obj_id}")

        return

    # QA Display (when QA pair exists) - optimized layout
    st.header("❓ Review This QA")

    # Show current progress
    _render_qa_progress()

    st.divider()

    # Question, answer, and additional info
    _render_qa_content(qa_pair, obj)

    st.divider()

    # Check if already annotated - show prominently
    _render_annotation_status(qa_pair)

    # Human annotation buttons
    _render_annotation_buttons(st.session_state.get("auto_advance", True))


def render_navigation():
    """Render navigation controls - simplified."""
    st.subheader("🧭 Quick Nav")

    # Skip buttons for faster navigation
    if st.button(
        "⏭️ Skip to Next Document",
        width="stretch",
        key="skip_doc",
        use_container_width=True,
    ):
        state.next_document()
        st.rerun()

    st.divider()

    # Manual navigation (collapsed by default)
    with st.expander("🎯 Manual Navigation", expanded=False):
        # Document navigation
        st.caption("📄 Document")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Prev", width="stretch", key="prev_doc"):
                state.prev_document()
                st.rerun()
        with col2:
            if st.button("Next ➡️", width="stretch", key="next_doc"):
                state.next_document()
                st.rerun()

        st.divider()

        # Object navigation
        st.caption("📦 Object")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Prev", width="stretch", key="prev_obj"):
                state.prev_object()
                st.rerun()
        with col2:
            if st.button("Next ➡️", width="stretch", key="next_obj"):
                state.next_object()
                st.rerun()

        st.divider()

        # QA navigation within object
        st.caption("❓ QA Pair")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬆️ Prev", width="stretch", key="prev_qa"):
                state.prev_qa()
                st.rerun()
        with col2:
            if st.button("Next ⬇️", width="stretch", key="next_qa"):
                state.next_qa()
                st.rerun()


def _render_image_display(image, obj, obj_id, show_cropped):
    """Render the main image display."""
    # Set reasonable max height to avoid scrolling (fits most screens)
    THUMBNAIL_HEIGHT = 600  # Increased slightly for better visibility

    # Show cropped or full image
    try:
        if obj and show_cropped and (obj.bbox or obj.polygon):
            cropped = crop_object_region(image, obj, padding=30)
            img_display = cropped
            caption = f"Object {obj_id} (Cropped)"
        else:
            # Use PIL to render bbox (obj can be None for empty documents)
            # max_height ensures image doesn't require scrolling
            img_display = render_image_with_bbox_pil(
                image, obj, highlight=True, max_height=THUMBNAIL_HEIGHT, max_width=800
            )
            if obj_id == -1:
                caption = "Document (No objects detected)"
            else:
                caption = f"Object {obj_id}"

        # Display thumbnail - width="stretch" makes it fill the column
        st.image(img_display, caption=caption, width="stretch")
    except Exception as e:
        st.error(f"⚠️ Failed to display image: {str(e)[:100]}")
        st.info("💡 This image may be corrupted. Try navigating to the next document.")


def _render_bbox_info(obj):
    """Render bounding box/polygon information."""
    if obj:
        info_parts = []
        if obj.bbox:
            info_parts.append(
                f"📦 ({obj.bbox[0]:.0f},{obj.bbox[1]:.0f})→({obj.bbox[2]:.0f},{obj.bbox[3]:.0f})"
            )
        if obj.polygon:
            info_parts.append(f"🔷 {len(obj.polygon)}pts")
        if obj.score:
            info_parts.append(f"✨ {obj.score:.0%}")

        if info_parts:
            st.caption(" | ".join(info_parts))
    else:
        st.caption("ℹ️ No text objects detected in this document")


def _render_fullsize_modal(image, obj, obj_id):
    """Render full-size image modal."""
    if "show_fullsize_modal" not in st.session_state:
        st.session_state.show_fullsize_modal = False

    if st.session_state.show_fullsize_modal:
        with st.expander("🔍 Full Size View", expanded=True):
            # Render larger version
            img_full = render_image_with_bbox_pil(
                image, obj, highlight=True, max_height=1600
            )
            st.image(img_full, width="stretch", caption=f"Object {obj_id} (Full Size)")

            if st.button("✖️ Close", key="close_modal"):
                st.session_state.show_fullsize_modal = False
                st.rerun()


def render_image_panel():
    """Render image with bounding box."""
    current = state.get_current_qa()

    if not current:
        return

    obj_id, qa_idx, qa_pair, obj, image = current

    # Check if image is None or corrupted
    if image is None:
        st.subheader("🖼️ Image")
        st.error("❌ Image could not be loaded (corrupted or missing)")
        return

    st.subheader("🖼️ Image")

    # Image view options - just crop checkbox and size info
    col1, col2 = st.columns([1, 1])
    with col1:
        show_cropped = st.checkbox(
            "🔍 Crop to Object",
            value=st.session_state.show_cropped,
            key="cropped_toggle",
        )
        st.session_state.show_cropped = show_cropped

    with col2:
        # Original size info
        st.caption(f"📐 {image.width}×{image.height}")

    # Show cropped or full image
    _render_image_display(image, obj, obj_id, show_cropped)

    # View Large button below image
    if st.button(
        "🔍 View Large Image",
        width="stretch",
        key="view_large",
        use_container_width=True,
    ):
        st.session_state.show_fullsize_modal = True

    # Show bbox/polygon info - compact one-liner (only if obj exists)
    _render_bbox_info(obj)

    # Modal for full-size view
    _render_fullsize_modal(image, obj, obj_id)


def main():
    """Main application."""
    # Page config
    st.set_page_config(
        page_title="QA Annotation Tool",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize state
    state.initialize_state()

    # Render sidebar
    render_sidebar()

    # Main content
    if not st.session_state.data_manager:
        st.info("👈 Please load data from the sidebar to start annotating.")
        st.markdown(
            """
            ## Welcome to QA Annotation Tool! 📝

            This tool helps you manually annotate and verify QA pairs generated from documents.

            ### How to use:
            1. **Load Data**: The sidebar is pre-filled with settings from `config.yml`:
               - Data Directory: `config.preprocess.output_dir` (JSON files)
               - Image Directory: `config.preprocess.input_dir` (images)
               - OCR Processor: `config.preprocess.processor` (default filter)
               - Choose processor filter to load only specific OCR outputs
               - Click "📂 Load Data" to start
            2. **Review**: Look at the document image, bounding box, and QA pair
            3. **Check Verifiers**: See what automatic verifiers think
            4. **Annotate**: Click "Approve" or "Reject" with optional explanation
            5. **Navigate**: Use navigation buttons to move between QA pairs

            ### Features:
            - ✅ Visual bounding box/polygon highlighting
            - 🔍 Click "View Large" to see full-size image
            - 🤖 Shows automatic verifier results
            - 📊 Real-time progress tracking
            - 💾 Auto-saves to JSON files
            - 🔧 Filter by OCR processor (paddleocr, easyocr, pdfplumber)
            - ⚙️ Auto-loads config from `config.yml`
            """
        )
        return

    # Three-column layout: Image (compact) | QA Info (larger) | Navigation (compact)
    # Now that image is small thumbnail, give more space to QA content
    col1, col2, col3 = st.columns([2, 3, 1])

    with col1:
        render_image_panel()

    with col2:
        render_qa_panel()

    with col3:
        render_navigation()

    # Simple footer
    st.divider()
    st.caption(
        "💡 **Quick Tip**: Enable 'Auto-advance' for fastest annotation workflow!"
    )


if __name__ == "__main__":
    main()
