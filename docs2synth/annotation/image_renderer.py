"""Image rendering with bounding boxes for annotation tool."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from PIL import Image as PILImage
from PIL import ImageDraw

from docs2synth.preprocess.schema import DocumentObject

logger = logging.getLogger(__name__)


def render_image_with_bbox_plotly(
    image: PILImage.Image,
    obj: Optional[DocumentObject] = None,
    highlight: bool = True,
) -> go.Figure:
    """Render image with bounding box using Plotly.

    Args:
        image: PIL Image
        obj: Document object with bbox to highlight
        highlight: Whether to highlight the current object

    Returns:
        Plotly figure
    """
    # Convert PIL image to numpy array
    img_array = np.array(image)

    # Create figure
    fig = go.Figure()

    # Add image
    fig.add_trace(go.Image(z=img_array))

    # Add bounding box if object is provided
    if obj and highlight:
        # Prefer polygon if available (more accurate), otherwise use bbox
        if obj.polygon:
            # Draw polygon (quadrilateral)
            polygon_points = obj.polygon
            # Close the polygon by adding the first point at the end
            x_coords = [p[0] for p in polygon_points] + [polygon_points[0][0]]
            y_coords = [p[1] for p in polygon_points] + [polygon_points[0][1]]

            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="lines",
                    line=dict(color="red", width=3),
                    fill="toself",
                    fillcolor="rgba(255, 0, 0, 0.1)",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            # Use first point for label position
            label_x, label_y = polygon_points[0]
        elif obj.bbox:
            # Fall back to axis-aligned rectangle
            bbox = obj.bbox
            x0, y0, x1, y1 = bbox

            # Draw rectangle
            fig.add_shape(
                type="rect",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                line=dict(color="red", width=3),
                fillcolor="rgba(255, 0, 0, 0.1)",
            )

            label_x, label_y = x0, y0
        else:
            # No bbox or polygon
            label_x = label_y = None

        # Add object ID label - place OUTSIDE bbox
        if label_x is not None and label_y is not None:
            # Place label above the bbox, outside the content area
            label_y_pos = label_y - 20  # Move further above

            fig.add_annotation(
                x=label_x,
                y=label_y_pos,
                text=f"Object {obj.object_id}",
                showarrow=False,
                font=dict(size=12, color="white"),
                bgcolor="red",
                bordercolor="red",
                borderwidth=1,
                xanchor="left",
                yanchor="bottom",
            )

    # Update layout
    fig.update_xaxes(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        range=[0, image.width],
    )
    fig.update_yaxes(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        range=[0, image.height],  # Normal y-axis: go.Image displays top-to-bottom
        scaleanchor="x",
        scaleratio=1,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=600,
        dragmode="pan",
    )

    return fig


def render_image_with_bbox_pil(
    image: PILImage.Image,
    obj: Optional[DocumentObject] = None,
    highlight: bool = True,
    max_width: int = 1200,
    max_height: int = 1600,
) -> PILImage.Image:
    """Render image with bounding box using PIL (fallback).

    Args:
        image: PIL Image
        obj: Document object with bbox/polygon to highlight
        highlight: Whether to highlight the current object
        max_width: Maximum width for display (will resize if larger)
        max_height: Maximum height for display (will resize if larger)

    Returns:
        PIL Image with bbox drawn (resized if necessary)
    """
    # Create a copy to avoid modifying original
    try:
        img_copy = image.copy()
    except (OSError, SyntaxError) as e:
        # Handle corrupted or broken image files
        logger.error(f"Failed to copy image: {e}")
        # Create a placeholder image
        img_copy = PILImage.new("RGB", (800, 600), color="lightgray")
        draw = ImageDraw.Draw(img_copy)
        error_msg = f"Image Error: {str(e)[:100]}"
        draw.text((50, 300), error_msg, fill="red")
        return img_copy

    # Calculate scale factor if image is too large
    scale_factor = 1.0
    if img_copy.width > max_width or img_copy.height > max_height:
        width_scale = max_width / img_copy.width
        height_scale = max_height / img_copy.height
        scale_factor = min(width_scale, height_scale)

        new_width = int(img_copy.width * scale_factor)
        new_height = int(img_copy.height * scale_factor)
        img_copy = img_copy.resize((new_width, new_height), PILImage.LANCZOS)

    if obj and highlight:
        draw = ImageDraw.Draw(img_copy)

        # Prefer polygon if available (more accurate for rotated text)
        if obj.polygon:
            # Draw polygon (quadrilateral)
            polygon_points = [
                (p[0] * scale_factor, p[1] * scale_factor) for p in obj.polygon
            ]
            draw.polygon(polygon_points, outline="red", width=3)

            # Use first point for label position
            label_x, label_y = polygon_points[0]
        elif obj.bbox:
            # Fall back to axis-aligned rectangle
            bbox = obj.bbox
            x0, y0, x1, y1 = [coord * scale_factor for coord in bbox]

            # Draw rectangle
            draw.rectangle(
                [x0, y0, x1, y1],
                outline="red",
                width=3,
            )

            label_x, label_y = x0, y0
        else:
            return img_copy

        # Draw label - place OUTSIDE bbox at top-left corner
        label_text = f"Object {obj.object_id}"

        # Calculate label position: place above and to the left of bbox
        # Try to place above the bbox first
        text_width = len(label_text) * 7 + 4
        text_height = 16
        label_offset = 5  # Offset from bbox edge

        # Place label above the bbox
        label_x_pos = max(0, label_x - label_offset)
        label_y_pos = max(0, label_y - text_height - label_offset)

        # If too close to top edge, place it to the left instead
        if label_y_pos < 5:
            label_y_pos = max(0, label_y)
            label_x_pos = max(0, label_x - text_width - label_offset)

        # If still too close to left edge, place it to the right
        if label_x_pos < 5:
            label_x_pos = min(img_copy.width - text_width, label_x + label_offset)

        # Draw semi-transparent background for better readability
        label_bbox = [
            label_x_pos,
            label_y_pos,
            min(img_copy.width, label_x_pos + text_width),
            min(img_copy.height, label_y_pos + text_height),
        ]
        draw.rectangle(label_bbox, fill="red")
        draw.text((label_x_pos + 2, label_y_pos + 2), label_text, fill="white")

    return img_copy


def crop_object_region(
    image: PILImage.Image,
    obj: DocumentObject,
    padding: int = 20,
) -> PILImage.Image:
    """Crop image to show only the object region.

    Args:
        image: PIL Image
        obj: Document object with bbox
        padding: Padding around bbox in pixels

    Returns:
        Cropped PIL Image
    """
    if not obj.bbox:
        return image

    bbox = obj.bbox
    x0, y0, x1, y1 = bbox

    # Add padding
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(image.width, x1 + padding)
    y1 = min(image.height, y1 + padding)

    return image.crop((x0, y0, x1, y1))
