"""Retriever training command-line interface commands.

This module provides CLI commands for training retriever models using
various training configurations (standard, layout, pretraining, etc.).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import click

from docs2synth.retriever import (
    pretrain_layout,
    train,
    train_layout,
    train_layout_coarse_grained,
    train_layout_gemini,
)
from docs2synth.utils import get_logger

logger = get_logger(__name__)


def _apply_run_id_to_path(
    path_str: str, run_id: Optional[str], default_filename: Optional[str] = None
) -> Path:
    """Resolve a run-scoped path using the configured run_id.

    Args:
        path_str: Base path string from config.
        run_id: Optional run identifier to scope artifacts.
        default_filename: Filename to append when the base path is a directory.

    Returns:
        Path: Resolved Path with run_id applied when provided.
    """

    base_path = Path(path_str)
    if not run_id:
        return base_path

    if "{run_id}" in path_str:
        return Path(path_str.format(run_id=run_id))

    if base_path.suffix:
        scoped_name = f"{base_path.stem}_{run_id}{base_path.suffix}"
        return base_path.with_name(scoped_name)

    scoped_path = base_path / run_id
    if default_filename:
        scoped_path = scoped_path / default_filename
    return scoped_path


def _load_model(
    resume: Optional[Path],
    model_path: Optional[Path],
    base_model: str,
) -> Any:
    """Load model from checkpoint, model path, or create custom QA model."""
    import torch

    from docs2synth.retriever.model import create_model_for_qa

    if resume:
        # Resume from checkpoint takes priority
        click.echo(f"  Resuming from checkpoint: {resume}")
        checkpoint = torch.load(resume, map_location="cpu")
        if "model_state_dict" in checkpoint:
            # Checkpoint contains state dict, need to create model first
            if model_path and model_path.exists():
                # Load complete model from path
                if model_path.is_file():
                    model = torch.load(model_path, map_location="cpu")
                else:
                    raise FileNotFoundError(f"Model file does not exist: {model_path}")
            else:
                # Create fresh model and load weights
                click.echo(f"  Creating custom QA model with base: {base_model}")
                model = create_model_for_qa(base_model_name=base_model)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Checkpoint is the model itself
            model = checkpoint
    elif model_path:
        # Load complete model from model path
        if model_path.is_file():
            model = torch.load(model_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
    else:
        # Create custom QA model from base LayoutLMv3
        click.echo(f"  Creating custom QA model from HuggingFace: {base_model}")
        model = create_model_for_qa(base_model_name=base_model)
    return model


def _load_training_data(data_path: Path) -> Any:
    """Load training data as DataLoader from pickle file.

    Args:
        data_path: Path to preprocessed DataLoader pickle file

    Returns:
        DataLoader instance

    Raises:
        ValueError: If data_path is not a valid pickle file
    """
    import pickle

    if not data_path.is_file():
        raise ValueError(
            f"Data path must be a pickle file (.pkl/.pickle), got: {data_path}\n"
            "Please preprocess your data first:\n"
            "  docs2synth retriever preprocess --json-dir <dir> --image-dir <dir> --output <file.pkl>"
        )

    if data_path.suffix not in [".pkl", ".pickle"]:
        raise ValueError(
            f"Data path must be a pickle file (.pkl/.pickle), got: {data_path.suffix}\n"
            "Please preprocess your data first:\n"
            "  docs2synth retriever preprocess --json-dir <dir> --image-dir <dir> --output <file.pkl>"
        )

    # Load pickled DataLoader
    click.echo(f"  Loading preprocessed DataLoader from: {data_path}")
    with open(data_path, "rb") as f:
        train_dataloader = pickle.load(f)

    return train_dataloader


def _get_training_function(mode: str) -> Any:
    """Get training function based on mode."""
    mode_lower = mode.lower()
    if mode_lower == "standard":
        return train
    elif mode_lower == "layout":
        return train_layout
    elif mode_lower == "layout-gemini":
        return train_layout_gemini
    elif mode_lower == "layout-coarse-grained":
        return train_layout_coarse_grained
    elif mode_lower == "pretrain-layout":
        return pretrain_layout
    else:
        raise ValueError(f"Unknown training mode: {mode}")


def _get_evaluation_function(mode: str) -> Optional[Any]:
    """Get evaluation function based on mode."""
    from docs2synth.retriever import evaluate, evaluate_layout

    mode_lower = mode.lower()
    if mode_lower == "standard":
        return evaluate
    elif mode_lower in ["layout", "layout-gemini", "layout-coarse-grained"]:
        return evaluate_layout
    elif mode_lower == "pretrain-layout":
        return None  # No evaluation for pretraining
    else:
        return None


def _run_training_loop(
    model: Any,
    train_dataloader: Any,
    train_func: Any,
    mode_lower: str,
    lr: float,
    epochs: int,
    start_epoch: int,
    save_every: Optional[int],
    output_dir: Path,
    val_dataloader: Optional[Any] = None,
    eval_func: Optional[Any] = None,
    device_str: Optional[str] = None,
) -> float:
    """Run training loop and return best ANLS score."""
    import warnings

    import torch

    # Suppress transformers FutureWarning about device argument
    warnings.filterwarnings(
        "ignore", category=FutureWarning, module="transformers.modeling_utils"
    )

    # Device selection: manual override or auto-detect
    if device_str:
        device = torch.device(device_str)
        click.echo(f"  Device: {device} (manual override)")
    elif torch.cuda.is_available():
        # Auto-detect best available device
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            click.echo(f"  Found {num_gpus} GPUs, using GPU 0")
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        click.echo(f"  Device: {device} ({gpu_name})")
    else:
        device = torch.device("cpu")
        click.echo(f"  Device: {device}")

    model = model.to(device)

    best_anls = 0.0
    for epoch in range(start_epoch, epochs):
        click.echo(
            click.style(
                f"\nEpoch {epoch + 1}/{epochs}",
                fg="blue",
                bold=True,
            )
        )

        # Train
        if mode_lower == "pretrain-layout":
            predict_entities, target_entities, total_loss = train_func(
                model, train_dataloader, lr
            )
            click.echo(
                click.style(
                    f"  Loss: {total_loss:.4f}",
                    fg="green",
                )
            )
            # For pretraining, save every epoch or as specified
            if save_every is None or (epoch + 1) % save_every == 0:
                checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pth"
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "loss": total_loss,
                    },
                    checkpoint_path,
                )
                click.echo(f"  Saved checkpoint: {checkpoint_path}")
        else:
            (
                average_anls,
                average_loss,
                pred_texts,
                gt_texts,
                pred_entities,
                target_entities,
            ) = train_func(model, train_dataloader, lr)
            click.echo(
                click.style(
                    f"  Train ANLS: {average_anls:.4f} | Train Loss: {average_loss:.4f}",
                    fg="green",
                )
            )
            best_anls = max(best_anls, average_anls)

            # Run validation if validation data is provided
            if val_dataloader is not None and eval_func is not None:
                (
                    val_anls,
                    val_loss,
                    val_pred_texts,
                    val_gt_texts,
                    val_pred_entities,
                    val_target_entities,
                ) = eval_func(model, val_dataloader)
                click.echo(
                    click.style(
                        f"  Val ANLS: {val_anls:.4f} | Val Loss: {val_loss:.4f}",
                        fg="cyan",
                    )
                )
                # Update best_anls based on validation if available
                best_anls = max(best_anls, val_anls)

            # Save checkpoint
            if save_every is None or (epoch + 1) % save_every == 0:
                checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pth"
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "average_anls": average_anls,
                        "average_loss": average_loss,
                    },
                    checkpoint_path,
                )
                click.echo(f"  Saved checkpoint: {checkpoint_path}")

    return best_anls


@click.group("retriever")
@click.pass_context
def retriever_group(ctx: click.Context) -> None:
    """Retriever training and evaluation commands."""
    pass


@retriever_group.command("train")
@click.option(
    "--model-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to existing model file or directory (optional, for continuing training)",
)
@click.option(
    "--base-model",
    type=str,
    default="microsoft/layoutlmv3-base",
    help="HuggingFace model name to use as base (default: microsoft/layoutlmv3-base). "
    "Only used if --model-path is not provided.",
)
@click.option(
    "--data-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to preprocessed training data (.pkl/.pickle file). "
    "If not provided, reads from config.yml (retriever.preprocessed_data_path). "
    "Run 'docs2synth retriever preprocess' first to create this file.",
)
@click.option(
    "--val-data-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to validation data (preprocessed DataLoader pickle). Optional.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to save trained model (defaults to config.data.models_dir)",
)
@click.option(
    "--mode",
    type=click.Choice(
        [
            "standard",
            "layout",
            "layout-gemini",
            "layout-coarse-grained",
            "pretrain-layout",
        ],
        case_sensitive=False,
    ),
    default="standard",
    help="Training mode: standard, layout, layout-gemini, layout-coarse-grained, or pretrain-layout",
)
@click.option(
    "--lr",
    type=float,
    default=None,
    help="Learning rate. If not provided, reads from config.yml (retriever.learning_rate, default: 1e-5)",
)
@click.option(
    "--epochs",
    type=int,
    default=None,
    help="Number of training epochs. If not provided, reads from config.yml (retriever.epochs, default: 10)",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Batch size (if not specified, uses DataLoader's batch_size)",
)
@click.option(
    "--save-every",
    type=int,
    default=None,
    help="Save checkpoint every N epochs. If not provided, reads from config.yml (retriever.save_every)",
)
@click.option(
    "--resume",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to checkpoint to resume training from",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Device to use for training (cpu, cuda, cuda:0, cuda:1, etc.). If not specified, auto-detects GPU.",
)
@click.pass_context
def retriever_train(  # noqa: C901
    ctx: click.Context,
    model_path: Optional[Path],
    base_model: str,
    data_path: Optional[Path],
    val_data_path: Optional[Path],
    output_dir: Optional[Path],
    mode: str,
    lr: Optional[float],
    epochs: Optional[int],
    batch_size: Optional[int],
    save_every: Optional[int],
    resume: Optional[Path],
    device: Optional[str],
) -> None:
    """Train a retriever model.

    This command trains a retriever model using annotated data. The training
    mode determines which training function to use:

    \b
    - standard: Standard training with entity retrieval and span-based QA
    - layout: Training with grid representations for layout pretraining
    - layout-gemini: Gemini variant with grid representations
    - layout-coarse-grained: Coarse-grained training (entity loss only)
    - pretrain-layout: Layout pretraining using grid embeddings

    You can either:
    1. Start from a HuggingFace base model (default: microsoft/layoutlmv3-base)
    2. Continue training from an existing model (--model-path)
    3. Resume from a checkpoint (--resume)

    Examples:
        # Start training from HuggingFace base model
        # (data path auto-loaded from config.yml: preprocess.output_dir)
        docs2synth retriever train --mode standard --lr 1e-5 --epochs 10

        # Specify data path explicitly
        docs2synth retriever train --data-path ./data/processed/train.pkl \\
            --mode standard --lr 1e-5 --epochs 10

        # Continue training from existing model
        docs2synth retriever train --model-path ./models/my_model.pth \\
            --mode standard --lr 1e-5

        # Use a different base model
        docs2synth retriever train --base-model microsoft/layoutlmv3-large \\
            --mode layout --lr 1e-5

        # Resume from checkpoint
        docs2synth retriever train --resume ./models/checkpoint_epoch_5.pth \\
            --mode standard
    """
    try:
        import torch
    except ImportError as e:
        click.echo(
            click.style(
                f"✗ Error: Failed to import PyTorch: {e}",
                fg="red",
            ),
            err=True,
        )
        if "libnccl" in str(e):
            click.echo(
                click.style(
                    "\nThis error indicates PyTorch is installed but missing the NCCL library.\n"
                    "Solutions:\n"
                    "  1. Install CPU-only PyTorch: pip uninstall torch && pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                    "  2. Install NCCL library: sudo apt-get install libnccl2\n"
                    "  3. Set environment variables: export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1",
                    fg="yellow",
                ),
                err=True,
            )
        else:
            click.echo(
                click.style(
                    "\nPyTorch is not installed or cannot be imported.\n"
                    "Install with: pip install torch",
                    fg="yellow",
                ),
                err=True,
            )
        sys.exit(1)

    try:
        cfg = ctx.obj.get("config")

        run_id = cfg.get("retriever.run_id")

        # Resolve data path from config if not provided
        if data_path is None:
            retriever_data_str = cfg.get("retriever.preprocessed_data_path")
            preprocess_output_str = cfg.get("preprocess.output_dir")

            if retriever_data_str:
                data_path = _apply_run_id_to_path(
                    retriever_data_str,
                    run_id,
                    default_filename="preprocessed_train.pkl",
                )
                click.echo(f"  Using data from config: {data_path}")
            elif preprocess_output_str:
                data_path = Path(preprocess_output_str)
                click.echo(f"  Using data from config: {data_path}")
            else:
                click.echo(
                    click.style(
                        "✗ Error: --data-path is required, or set retriever.preprocessed_data_path in config.yml",
                        fg="red",
                    ),
                    err=True,
                )
                sys.exit(1)
        else:
            data_path = Path(data_path)

        # Resolve hyperparameters from config if not provided
        if lr is None:
            lr_value = cfg.get("retriever.learning_rate", 1e-5)
            lr = float(lr_value)  # Ensure it's a float
            click.echo(f"  Using learning rate from config: {lr}")

        if epochs is None:
            epochs_value = cfg.get("retriever.epochs", 10)
            epochs = int(epochs_value)  # Ensure it's an int
            click.echo(f"  Using epochs from config: {epochs}")

        if save_every is None:
            save_every_value = cfg.get("retriever.save_every")
            if save_every_value:
                save_every = int(save_every_value)  # Ensure it's an int
                click.echo(f"  Using save_every from config: {save_every}")

        # Validate data path exists
        if not data_path.exists():
            click.echo(
                click.style(
                    f"✗ Error: Data path does not exist: {data_path}",
                    fg="red",
                ),
                err=True,
            )
            sys.exit(1)

        if output_dir is None:
            # Use retriever checkpoint directory from config
            checkpoint_dir_str = cfg.get("retriever.checkpoint_dir")
            if checkpoint_dir_str:
                output_dir = _apply_run_id_to_path(checkpoint_dir_str, run_id)
            else:
                # Fallback to default models directory
                output_dir = Path("./models/retriever/checkpoints")
                if run_id:
                    output_dir = output_dir / run_id
            click.echo(f"  Using checkpoint directory: {output_dir}")
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        click.echo(
            click.style(
                f"Training retriever model in '{mode}' mode...",
                fg="blue",
                bold=True,
            )
        )
        if model_path:
            click.echo(f"  Model path: {model_path}")
        else:
            click.echo(f"  Base model: {base_model}")
        click.echo(f"  Data: {data_path}")
        click.echo(f"  Output: {output_dir}")
        click.echo(f"  Learning rate: {lr}")
        click.echo(f"  Epochs: {epochs}")

        # Load model
        click.echo(click.style("Loading model...", fg="yellow"))
        try:
            model = _load_model(resume, model_path, base_model)
        except Exception as e:
            click.echo(
                click.style(
                    f"✗ Error: Failed to load model: {e}",
                    fg="red",
                ),
                err=True,
            )
            logger.exception("Model loading failed")
            sys.exit(1)

        # Load preprocessed DataLoader
        click.echo(click.style("Loading training data...", fg="yellow"))
        try:
            train_dataloader = _load_training_data(data_path)
        except Exception as e:
            click.echo(
                click.style(
                    f"✗ Error: Failed to load training data: {e}",
                    fg="red",
                ),
                err=True,
            )
            logger.exception("Data loading failed")
            sys.exit(1)

        # Note: Batch size is fixed at preprocessing time
        if batch_size is not None and hasattr(train_dataloader, "batch_size"):
            if train_dataloader.batch_size != batch_size:
                click.echo(
                    click.style(
                        f"⚠ Warning: Requested batch_size={batch_size} but DataLoader has batch_size={train_dataloader.batch_size}. "
                        "Using DataLoader's batch_size. Recreate the pickle file with desired batch_size.",
                        fg="yellow",
                    )
                )

        # Load validation data if provided
        val_dataloader = None
        if val_data_path is not None:
            click.echo(click.style("Loading validation data...", fg="yellow"))
            try:
                val_dataloader = _load_training_data(val_data_path)
                # Note: Same warning applies for validation data
                if batch_size is not None and hasattr(val_dataloader, "batch_size"):
                    if val_dataloader.batch_size != batch_size:
                        click.echo(
                            click.style(
                                f"⚠ Warning: Validation DataLoader has batch_size={val_dataloader.batch_size}. Using that value.",
                                fg="yellow",
                            )
                        )
            except Exception as e:
                click.echo(
                    click.style(
                        f"✗ Error: Failed to load validation data: {e}",
                        fg="red",
                    ),
                    err=True,
                )
                logger.exception("Validation data loading failed")
                sys.exit(1)

        # Get start epoch from checkpoint if resuming
        start_epoch = 0
        if resume:
            try:
                checkpoint = torch.load(resume, map_location="cpu")
                start_epoch = checkpoint.get("epoch", 0)
                if "optimizer_state_dict" in checkpoint:
                    logger.info(
                        "Note: Optimizer state found but not restored (new optimizer created)"
                    )
            except Exception as e:
                logger.warning(f"Could not extract epoch from checkpoint: {e}")

        # Select training function based on mode
        try:
            train_func = _get_training_function(mode)
        except ValueError as e:
            click.echo(
                click.style(f"✗ Error: {e}", fg="red"),
                err=True,
            )
            sys.exit(1)

        mode_lower = mode.lower()

        # Get evaluation function if validation data is provided
        eval_func = None
        if val_dataloader is not None:
            eval_func = _get_evaluation_function(mode)
            if eval_func is None:
                click.echo(
                    click.style(
                        "⚠ Warning: Validation data provided but no evaluation function available for this mode",
                        fg="yellow",
                    )
                )

        # Training loop
        best_anls = _run_training_loop(
            model,
            train_dataloader,
            train_func,
            mode_lower,
            lr,
            epochs,
            start_epoch,
            save_every,
            output_dir,
            val_dataloader,
            eval_func,
            device,
        )

        # Save final model
        import torch

        # Use configured model path or default
        final_model_path_str = cfg.get("retriever.model_path")
        if final_model_path_str:
            final_model_path = _apply_run_id_to_path(
                final_model_path_str, run_id, default_filename="final_model.pth"
            )
        else:
            final_model_path = Path("./models/retriever/final_model.pth")
            if run_id:
                final_model_path = (
                    final_model_path.parent / run_id / final_model_path.name
                )

        # Ensure model directory exists
        final_model_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "epoch": epochs,
                "model_state_dict": model.state_dict(),
                "best_anls": best_anls if mode_lower != "pretrain-layout" else None,
            },
            final_model_path,
        )

        click.echo(
            click.style(
                "\n✓ Training complete!",
                fg="green",
                bold=True,
            )
        )
        click.echo(f"  Final model saved: {final_model_path}")
        if mode_lower != "pretrain-layout":
            click.echo(f"  Best ANLS: {best_anls:.4f}")
        click.echo(f"  Checkpoints saved in: {output_dir}")

    except Exception as e:
        logger.exception("Retriever training failed")
        click.echo(click.style(f"✗ Error: {e}", fg="red"), err=True)
        sys.exit(1)


@retriever_group.command("preprocess")
@click.option(
    "--json-dir",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Directory containing processed JSON files. "
    "If not provided, reads from config.yml (preprocess.output_dir)",
)
@click.option(
    "--image-dir",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Directory containing document images. "
    "If not provided, reads from config.yml (preprocess.input_dir)",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path for the preprocessed DataLoader pickle file. "
    "If not provided, generates default path in data directory (e.g., ./data/preprocessed_train.pkl)",
)
@click.option(
    "--processor",
    type=str,
    default="docling",
    help="Processor name to filter JSON files (default: docling). "
    "Will load files matching *_{processor}.json pattern",
)
@click.option(
    "--batch-size",
    type=int,
    default=8,
    help="Batch size for DataLoader (default: 8)",
)
@click.option(
    "--max-length",
    type=int,
    default=512,
    help="Maximum sequence length for tokenization (default: 512)",
)
@click.option(
    "--num-objects",
    type=int,
    default=50,
    help="Maximum number of objects per document (default: 50)",
)
@click.option(
    "--require-all-verifiers/--no-require-all-verifiers",
    default=True,
    help="Whether all verifiers must respond 'Yes' (default: True)",
)
@click.pass_context
def retriever_preprocess(
    ctx: click.Context,
    json_dir: Optional[Path],
    image_dir: Optional[Path],
    output: Optional[Path],
    processor: str,
    batch_size: int,
    max_length: int,
    num_objects: int,
    require_all_verifiers: bool,
) -> None:
    """Preprocess JSON QA pairs into training DataLoader format.

    This command converts verified QA pairs from JSON files into the tensor
    format required by the training functions, and saves the result as a
    pickle file for efficient training.

    Stage 1 (MVP): Implements basic fields with simplified versions for complex features.
    - Basic fields: Uses LayoutLMv3 processor (input_ids, attention_mask, bbox, pixel_values)
    - Complex fields: Simplified or placeholder implementations

    Examples:
        # Preprocess using all config defaults
        docs2synth retriever preprocess

        # Specify custom output path
        docs2synth retriever preprocess \\
            --output ./data/train_dataloader.pkl

        # Specify all parameters explicitly
        docs2synth retriever preprocess \\
            --json-dir ./data/processed/dev/ \\
            --image-dir ./data/datasets/docs2synth-dev/docs2synth-dev/images/ \\
            --output ./data/train_dataloader.pkl \\
            --processor docling \\
            --batch-size 8 \\
            --max-length 512

        # Use different processor
        docs2synth retriever preprocess \\
            --json-dir ./data/processed/dev/ \\
            --image-dir ./data/images/ \\
            --output ./data/train_paddleocr.pkl \\
            --processor paddleocr
    """
    try:
        from docs2synth.retriever.preprocess import create_preprocessed_dataloader

        cfg = ctx.obj.get("config")
        run_id = cfg.get("retriever.run_id")

        # Resolve json_dir from config if not provided
        if json_dir is None:
            json_dir_str = cfg.get("preprocess.output_dir")
            if json_dir_str:
                json_dir = Path(json_dir_str)
                click.echo(f"  Using JSON directory from config: {json_dir}")
            else:
                click.echo(
                    click.style(
                        "✗ Error: --json-dir is required, or set preprocess.output_dir in config.yml",
                        fg="red",
                    ),
                    err=True,
                )
                sys.exit(1)

        # Resolve image_dir from config if not provided
        if image_dir is None:
            image_dir_str = cfg.get("preprocess.input_dir")
            if image_dir_str:
                image_dir = Path(image_dir_str)
                click.echo(f"  Using image directory from config: {image_dir}")
            else:
                click.echo(
                    click.style(
                        "✗ Error: --image-dir is required, or set preprocess.input_dir in config.yml",
                        fg="red",
                    ),
                    err=True,
                )
                sys.exit(1)

        # Resolve output path from config if not provided
        if output is None:
            # Read from config
            output_str = cfg.get("retriever.preprocessed_data_path")
            if output_str:
                output = _apply_run_id_to_path(
                    output_str, run_id, default_filename="preprocessed_train.pkl"
                )
                click.echo(f"  Using output path from config: {output}")
            else:
                # Fallback to default path
                output = Path("./data/retriever/preprocessed_train.pkl")
                if run_id:
                    output = output.parent / run_id / output.name
                click.echo(f"  Using default output path: {output}")
        else:
            output = Path(output)

        # Ensure output directory exists
        output.parent.mkdir(parents=True, exist_ok=True)

        # Validate paths
        if not json_dir.exists():
            click.echo(
                click.style(
                    f"✗ Error: JSON directory does not exist: {json_dir}",
                    fg="red",
                ),
                err=True,
            )
            sys.exit(1)

        if not image_dir.exists():
            click.echo(
                click.style(
                    f"✗ Error: Image directory does not exist: {image_dir}",
                    fg="red",
                ),
                err=True,
            )
            sys.exit(1)

        click.echo(
            click.style(
                "Preprocessing JSON QA pairs → DataLoader pickle...",
                fg="blue",
                bold=True,
            )
        )
        click.echo(f"  JSON directory: {json_dir}")
        click.echo(f"  Image directory: {image_dir}")
        click.echo(f"  Output: {output}")
        click.echo(f"  Processor: {processor}")
        click.echo(f"  Batch size: {batch_size}")
        click.echo(f"  Max sequence length: {max_length}")
        click.echo(f"  Max objects: {num_objects}")
        click.echo(f"  Require all verifiers: {require_all_verifiers}")

        # Create preprocessed dataloader
        dataloader = create_preprocessed_dataloader(
            json_dir=json_dir,
            image_dir=image_dir,
            output_path=output,
            processor_name=processor,
            batch_size=batch_size,
            max_length=max_length,
            num_objects=num_objects,
            require_all_verifiers=require_all_verifiers,
        )

        click.echo(
            click.style(
                "\n✓ Preprocessing complete!",
                fg="green",
                bold=True,
            )
        )
        click.echo(f"  Saved: {output}")
        click.echo(f"  QA pairs: {len(dataloader.dataset)}")
        click.echo(f"  Batches: {len(dataloader)}")
        click.echo("\n  Use with:")
        click.echo(f"    docs2synth retriever train --data-path {output}")

    except Exception as e:
        logger.exception("Preprocessing failed")
        click.echo(click.style(f"✗ Error: {e}", fg="red"), err=True)
        sys.exit(1)
