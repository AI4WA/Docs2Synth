"""Tests for retriever model."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from docs2synth.retriever.model import (
    DEFAULT_DROPOUT_RATE,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_NUM_OBJECTS,
    LayoutLMForDocumentQA,
    create_model_for_qa,
)


@pytest.fixture
def mock_layoutlm_model():
    """Create a mock LayoutLMv3 model for testing."""
    mock_model = MagicMock()

    # Mock the forward pass to return proper structure
    mock_output = MagicMock()
    mock_output.last_hidden_state = torch.randn(
        2, 10, 768
    )  # [batch_size, seq_len, hidden_size]
    mock_model.return_value = mock_output

    return mock_model


def test_model_initialization(mock_layoutlm_model) -> None:
    """Test LayoutLMForDocumentQA initialization."""
    model = LayoutLMForDocumentQA(
        layoutlm_model=mock_layoutlm_model,
        hidden_size=768,
        num_objects=50,
        dropout_rate=0.1,
    )

    assert model.layoutlm == mock_layoutlm_model
    assert model.hidden_size == 768
    assert model.num_objects == 50
    assert isinstance(model.qa_outputs, nn.Linear)
    assert model.qa_outputs.in_features == 768
    assert model.qa_outputs.out_features == 2  # start and end logits


def test_model_initialization_with_defaults(mock_layoutlm_model) -> None:
    """Test model initialization uses default values."""
    model = LayoutLMForDocumentQA(layoutlm_model=mock_layoutlm_model)

    assert model.hidden_size == DEFAULT_HIDDEN_SIZE
    assert model.num_objects == DEFAULT_NUM_OBJECTS


def test_model_forward_pass(mock_layoutlm_model) -> None:
    """Test forward pass through the model."""
    # Setup mock model to return proper tensor
    mock_output = MagicMock()
    batch_size, seq_len, hidden_size = 2, 10, 768
    mock_output.last_hidden_state = torch.randn(batch_size, seq_len, hidden_size)
    mock_layoutlm_model.return_value = mock_output

    model = LayoutLMForDocumentQA(
        layoutlm_model=mock_layoutlm_model, hidden_size=hidden_size
    )

    # Create dummy inputs
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    bbox = torch.zeros(batch_size, seq_len, 4, dtype=torch.long)

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        pixel_values=pixel_values,
        bbox=bbox,
    )

    # Check outputs
    assert "start_logits" in outputs
    assert "end_logits" in outputs
    assert outputs["start_logits"].shape == (batch_size, seq_len)
    assert outputs["end_logits"].shape == (batch_size, seq_len)

    # Verify layoutlm was called with correct arguments
    mock_layoutlm_model.assert_called_once()
    call_kwargs = mock_layoutlm_model.call_args[1]
    assert "input_ids" in call_kwargs
    assert "attention_mask" in call_kwargs
    assert "token_type_ids" in call_kwargs
    assert "bbox" in call_kwargs
    assert "pixel_values" in call_kwargs
    assert call_kwargs["return_dict"] is True


def test_model_forward_with_optional_params(mock_layoutlm_model) -> None:
    """Test forward pass with optional compatibility parameters."""
    mock_output = MagicMock()
    batch_size, seq_len, hidden_size = 1, 5, 768
    mock_output.last_hidden_state = torch.randn(batch_size, seq_len, hidden_size)
    mock_layoutlm_model.return_value = mock_output

    model = LayoutLMForDocumentQA(
        layoutlm_model=mock_layoutlm_model, hidden_size=hidden_size
    )

    # Create inputs with optional parameters (should be ignored)
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    bbox = torch.zeros(batch_size, seq_len, 4, dtype=torch.long)

    # Extra parameters (kept for compatibility, should be ignored)
    encoded_token_objt_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    bert_cls = torch.randn(batch_size, hidden_size)
    visual_feat = torch.randn(batch_size, 50, 2048)
    norm_bbox = torch.zeros(batch_size, 50, 4)
    object_mask = torch.ones(batch_size, 50)
    positional_encoding = torch.randn(batch_size, seq_len, hidden_size)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        pixel_values=pixel_values,
        bbox=bbox,
        encoded_token_objt_ids=encoded_token_objt_ids,
        bert_cls=bert_cls,
        visual_feat=visual_feat,
        norm_bbox=norm_bbox,
        object_mask=object_mask,
        positional_encoding=positional_encoding,
    )

    # Should still work and return proper outputs
    assert "start_logits" in outputs
    assert "end_logits" in outputs


def test_model_output_shapes_different_batch_sizes(mock_layoutlm_model) -> None:
    """Test model handles different batch sizes correctly."""
    hidden_size = 768

    model = LayoutLMForDocumentQA(
        layoutlm_model=mock_layoutlm_model, hidden_size=hidden_size
    )

    for batch_size in [1, 4, 8]:
        seq_len = 12
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(batch_size, seq_len, hidden_size)
        mock_layoutlm_model.return_value = mock_output

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        bbox = torch.zeros(batch_size, seq_len, 4, dtype=torch.long)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            bbox=bbox,
        )

        assert outputs["start_logits"].shape == (batch_size, seq_len)
        assert outputs["end_logits"].shape == (batch_size, seq_len)


@patch("docs2synth.retriever.model.LayoutLMv3Model")
def test_create_model_for_qa(mock_layoutlmv3_class) -> None:
    """Test create_model_for_qa function."""
    # Mock the LayoutLMv3Model.from_pretrained
    mock_layoutlm = MagicMock()
    mock_layoutlm.config.hidden_size = 768
    mock_layoutlmv3_class.from_pretrained.return_value = mock_layoutlm

    model = create_model_for_qa(
        base_model_name="microsoft/layoutlmv3-base",
        num_objects=50,
        dropout_rate=0.1,
    )

    # Check that from_pretrained was called
    mock_layoutlmv3_class.from_pretrained.assert_called_once_with(
        "microsoft/layoutlmv3-base"
    )

    # Check model type and properties
    assert isinstance(model, LayoutLMForDocumentQA)
    assert model.hidden_size == 768
    assert model.num_objects == 50


@patch("docs2synth.retriever.model.LayoutLMv3Model")
def test_create_model_for_qa_with_defaults(mock_layoutlmv3_class) -> None:
    """Test create_model_for_qa uses default parameters."""
    mock_layoutlm = MagicMock()
    mock_layoutlm.config.hidden_size = 768
    mock_layoutlmv3_class.from_pretrained.return_value = mock_layoutlm

    model = create_model_for_qa()

    # Should use defaults
    mock_layoutlmv3_class.from_pretrained.assert_called_once_with(
        "microsoft/layoutlmv3-base"
    )
    assert model.num_objects == DEFAULT_NUM_OBJECTS


@patch("docs2synth.retriever.model.LayoutLMv3Model")
def test_create_model_for_qa_custom_base_model(mock_layoutlmv3_class) -> None:
    """Test create_model_for_qa with custom base model."""
    mock_layoutlm = MagicMock()
    mock_layoutlm.config.hidden_size = 1024
    mock_layoutlmv3_class.from_pretrained.return_value = mock_layoutlm

    model = create_model_for_qa(base_model_name="custom/layoutlm-model")

    mock_layoutlmv3_class.from_pretrained.assert_called_once_with(
        "custom/layoutlm-model"
    )
    assert model.hidden_size == 1024


def test_default_constants() -> None:
    """Test default constant values are reasonable."""
    assert DEFAULT_HIDDEN_SIZE == 768
    assert DEFAULT_NUM_OBJECTS == 50
    assert DEFAULT_DROPOUT_RATE == 0.1
    assert 0.0 < DEFAULT_DROPOUT_RATE < 1.0


def test_model_is_torch_module(mock_layoutlm_model) -> None:
    """Test model is a proper PyTorch module."""
    model = LayoutLMForDocumentQA(layoutlm_model=mock_layoutlm_model)
    assert isinstance(model, nn.Module)


def test_qa_outputs_layer_structure(mock_layoutlm_model) -> None:
    """Test qa_outputs layer has correct structure."""
    hidden_size = 512
    model = LayoutLMForDocumentQA(
        layoutlm_model=mock_layoutlm_model, hidden_size=hidden_size
    )

    assert isinstance(model.qa_outputs, nn.Linear)
    assert model.qa_outputs.in_features == hidden_size
    assert model.qa_outputs.out_features == 2  # For start and end
    assert model.qa_outputs.bias is not None  # Linear layer has bias by default
