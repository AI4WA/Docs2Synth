"""Retriever training command-line interface commands.

This module provides CLI commands for training retriever models using
various training configurations (standard, layout, pretraining, etc.).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _get_device_from_config(cfg: Any, device_override: Optional[str] = None) -> Any:
    """Get device from config or use override.

    Priority:
    1. CLI override (--device flag)
    2. Global device config
    3. Preprocess device config (deprecated, for backward compatibility)
    4. Auto-detect (CUDA if available, else CPU)

    Args:
        cfg: Configuration object
        device_override: Optional device string from CLI

    Returns:
        torch.device object
    """
    import torch

    # Priority 1: CLI override
    if device_override:
        device = torch.device(device_override)
        return device, f"{device_override} (CLI override)"

    # Priority 2: Global device config
    global_device = cfg.get("device")
    if global_device:
        device = torch.device(global_device)
        return device, f"{global_device} (from config)"

    # Priority 3: Preprocess device config (backward compatibility)
    preprocess_device = cfg.get("preprocess.device")
    if preprocess_device:
        device = torch.device(preprocess_device)
        return device, f"{preprocess_device} (from preprocess config, deprecated)"

    # Priority 4: Auto-detect
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        return device, f"cuda:0 (auto-detected: {gpu_name})"
    else:
        device = torch.device("cpu")
        return device, "cpu (auto-detected: no CUDA available)"


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
    device: Any = None,
    cfg: Any = None,
    device_override: Optional[str] = None,
) -> float:
    """Run training loop and return best ANLS score."""
    import warnings

    import torch

    # Suppress transformers FutureWarning about device argument
    warnings.filterwarnings(
        "ignore", category=FutureWarning, module="transformers.modeling_utils"
    )

    # Get device from config or override
    if device is None:
        device, device_desc = _get_device_from_config(cfg, device_override)
        click.echo(f"  Device: {device_desc}")
    else:
        # Device already provided (for backward compatibility)
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
            device=None,  # Will be determined from config
            cfg=cfg,
            device_override=device,
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


def _load_model_for_validation(model_path: Path) -> Any:
    """Load model for validation.

    Args:
        model_path: Path to model checkpoint

    Returns:
        Loaded model
    """
    import torch

    from docs2synth.retriever.model import create_model_for_qa

    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        trained_model = create_model_for_qa()
        trained_model.load_state_dict(checkpoint["model_state_dict"])
        click.echo(
            f"  Loaded from checkpoint (epoch: {checkpoint.get('epoch', 'N/A')})"
        )
    else:
        trained_model = checkpoint
        click.echo("  Loaded model directly")
    return trained_model


def _load_validation_data(data_path: Path) -> Any:
    """Load validation data from pickle.

    Args:
        data_path: Path to DataLoader pickle

    Returns:
        DataLoader instance
    """
    import pickle

    with open(data_path, "rb") as f:
        val_dataloader = pickle.load(f)
    click.echo(f"  Dataset size: {len(val_dataloader.dataset)}")
    click.echo(f"  Batch size: {val_dataloader.batch_size}")
    click.echo(f"  Number of batches: {len(val_dataloader)}")
    return val_dataloader


def _print_validation_summary(analysis: Dict[str, Any], pred_texts: List[str]) -> None:
    """Print validation results summary.

    Args:
        analysis: Analysis results dictionary
        pred_texts: List of predicted texts
    """
    # ANLS Score
    click.echo("\n" + click.style("ANLS Score:", bold=True))
    anls = analysis["anls"]
    click.echo(f"  Mean:            {anls['mean']:.4f}")
    click.echo(f"  Std:             {anls['std']:.4f}")
    click.echo(f"  Median:          {anls['median']:.4f}")
    click.echo(f"  Range:           [{anls['min']:.4f}, {anls['max']:.4f}]")
    perfect_pct = anls["perfect_matches"] / len(pred_texts) * 100
    zero_pct = anls["zero_matches"] / len(pred_texts) * 100
    click.echo(f"  Perfect matches: {anls['perfect_matches']} ({perfect_pct:.1f}%)")
    click.echo(f"  Zero matches:    {anls['zero_matches']} ({zero_pct:.1f}%)")

    # Entity Retrieval
    click.echo("\n" + click.style("Entity Retrieval:", bold=True))
    entity = analysis["entity"]
    click.echo(f"  Accuracy:        {entity['accuracy']:.4f}")
    click.echo(f"  Correct:         {entity['correct']} / {entity['total']}")

    # Prediction Length
    click.echo("\n" + click.style("Prediction Length:", bold=True))
    length = analysis["length"]
    click.echo(f"  Predicted mean:  {length['pred_mean']:.2f} words")
    click.echo(f"  Ground truth:    {length['gt_mean']:.2f} words")


def _print_sanity_checks(checks: Dict[str, Any], pred_texts: List[str]) -> None:
    """Print sanity check results.

    Args:
        checks: Sanity check results
        pred_texts: List of predicted texts
    """
    click.echo("\n" + click.style("=" * 70, fg="blue"))
    click.echo(click.style("SANITY CHECKS", fg="blue", bold=True))
    click.echo(click.style("=" * 70, fg="blue") + "\n")

    # Check 1: Empty predictions
    empty_ratio = checks["empty_predictions"] / len(pred_texts)
    if empty_ratio < 0.05:
        status = click.style("✅", fg="green")
        msg = "Good"
    elif empty_ratio < 0.2:
        status = click.style("⚠️ ", fg="yellow")
        msg = "Acceptable"
    else:
        status = click.style("❌", fg="red")
        msg = "Too many!"
    click.echo(
        f"{status} Empty predictions: {checks['empty_predictions']} ({empty_ratio*100:.1f}%) - {msg}"
    )

    # Check 2: Prediction diversity
    diversity_ratio = checks["unique_predictions"] / len(pred_texts)
    if diversity_ratio > 0.5:
        status = click.style("✅", fg="green")
        msg = "Good"
    elif diversity_ratio > 0.2:
        status = click.style("⚠️ ", fg="yellow")
        msg = "Acceptable"
    else:
        status = click.style("❌", fg="red")
        msg = "Too low!"
    click.echo(
        f"{status} Prediction diversity: {checks['unique_predictions']} unique ({diversity_ratio*100:.1f}%) - {msg}"
    )

    # Check 3: Entity diversity
    entity_diversity = checks["unique_entities"]
    if entity_diversity > 10:
        status = click.style("✅", fg="green")
        msg = "Good"
    elif entity_diversity > 5:
        status = click.style("⚠️ ", fg="yellow")
        msg = "Acceptable"
    else:
        status = click.style("❌", fg="red")
        msg = "Model may have collapsed!"
    click.echo(f"{status} Entity diversity: {entity_diversity} unique entities - {msg}")


def _print_worst_predictions(analysis: Dict[str, Any]) -> None:
    """Print worst predictions.

    Args:
        analysis: Analysis results dictionary
    """
    click.echo("\n" + click.style("=" * 70, fg="blue"))
    click.echo(click.style("WORST PREDICTIONS (Top 5)", fg="blue", bold=True))
    click.echo(click.style("=" * 70, fg="blue"))

    for i, error in enumerate(analysis["worst_predictions"][:5], 1):
        click.echo(f"\n{i}. ANLS: {error['anls']:.4f}")
        click.echo(f"   Predicted:     {error['predicted'][:80]}")
        click.echo(f"   Ground Truth:  {error['ground_truth'][:80]}")
        click.echo(f"   Entity: {error['entity_pred']} → {error['entity_target']}")


def _auto_discover_model_path(cfg: Any, run_id: Optional[str] = None) -> Optional[Path]:
    """Automatically discover model path from config or checkpoints.

    Args:
        cfg: Configuration object
        run_id: Optional run identifier

    Returns:
        Path to model file, or None if not found
    """
    # Try 1: Look for final model from config
    final_model_path_str = cfg.get("retriever.model_path")
    if final_model_path_str:
        final_model_path = _apply_run_id_to_path(
            final_model_path_str, run_id, default_filename="final_model.pth"
        )
        if final_model_path.exists():
            click.echo(f"  Auto-discovered model: {final_model_path}")
            return final_model_path

    # Try 2: Look for default final model path
    default_final_model = Path("./models/retriever/final_model.pth")
    if run_id:
        default_final_model = (
            default_final_model.parent / run_id / default_final_model.name
        )
    if default_final_model.exists():
        click.echo(f"  Auto-discovered model: {default_final_model}")
        return default_final_model

    # Try 3: Look for latest checkpoint in checkpoint directory
    checkpoint_dir_str = cfg.get("retriever.checkpoint_dir")
    if checkpoint_dir_str:
        checkpoint_dir = _apply_run_id_to_path(checkpoint_dir_str, run_id)
    else:
        checkpoint_dir = Path("./models/retriever/checkpoints")
        if run_id:
            checkpoint_dir = checkpoint_dir / run_id

    if checkpoint_dir.exists():
        # Find all checkpoint files
        checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        if checkpoints:
            # Sort by epoch number and get the latest
            def get_epoch_num(p: Path) -> int:
                try:
                    return int(p.stem.split("_")[-1])
                except (ValueError, IndexError):
                    return 0

            latest_checkpoint = max(checkpoints, key=get_epoch_num)
            click.echo(f"  Auto-discovered latest checkpoint: {latest_checkpoint}")
            return latest_checkpoint

    return None


@retriever_group.command("validate")
@click.option(
    "--model",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to trained model checkpoint (.pth file). If not provided, automatically loads from config or latest checkpoint.",
)
@click.option(
    "--data",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to validation DataLoader pickle (.pkl file). If not provided, reads from config.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for validation reports (defaults to ./models/retriever/{run_id}/validation_reports/)",
)
@click.option(
    "--mode",
    type=click.Choice(["standard", "layout"], case_sensitive=False),
    default="standard",
    help="Evaluation mode (default: standard)",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Device to use for evaluation (cpu, cuda, cuda:0, cuda:1, etc.). If not specified, auto-detects GPU.",
)
@click.pass_context
def retriever_validate(  # noqa: C901
    ctx: click.Context,
    model: Optional[Path],
    data: Optional[Path],
    output: Optional[Path],
    mode: str,
    device: Optional[str],
) -> None:
    """Validate retriever training results.

    This command performs comprehensive validation of a trained retriever model,
    including:
    - ANLS score analysis
    - Entity retrieval accuracy
    - Prediction diversity checks
    - Error analysis (worst predictions)
    - Training curve visualization

    Auto-discovery (when parameters not specified):

    Model path (--model):
    1. Config retriever.model_path
    2. Default: ./models/retriever/final_model.pth
    3. Latest checkpoint in retriever.checkpoint_dir

    Validation data (--data):
    1. Config retriever.validation_data_path
    2. Same directory as retriever.preprocessed_data_path (with 'val' instead of 'train')
    3. Config preprocess.output_dir
    4. Default: ./data/retriever/preprocessed_val.pkl

    Examples:
        # Auto-discover both model and data from config
        docs2synth retriever validate

        # Auto-discover model, specify data
        docs2synth retriever validate --data data/val.pkl

        # Specify both explicitly
        docs2synth retriever validate \\
            --model models/retriever/final_model.pth \\
            --data data/val.pkl

        # With custom output directory
        docs2synth retriever validate \\
            --data data/val.pkl \\
            --output ./my_custom_reports/

        # For layout mode
        docs2synth retriever validate --mode layout
    """
    from docs2synth.retriever.validation import TrainingValidator

    try:
        cfg = ctx.obj.get("config")
        run_id = cfg.get("retriever.run_id")

        # Auto-discover validation data path if not provided
        if data is None:
            click.echo(
                click.style(
                    "No data path provided, attempting auto-discovery...", fg="yellow"
                )
            )

            searched_paths = []

            # Try 1: Explicit validation data path from config
            val_data_str = cfg.get("retriever.validation_data_path")
            if val_data_str:
                candidate = _apply_run_id_to_path(
                    val_data_str, run_id, default_filename="preprocessed_val.pkl"
                )
                searched_paths.append(
                    ("Config retriever.validation_data_path", candidate)
                )
                if candidate.exists():
                    data = candidate
                    click.echo(f"  ✓ Found validation data: {data}")

            # Try 2: Look in the same directory as training data (preprocessed_data_path)
            if data is None:
                train_data_str = cfg.get("retriever.preprocessed_data_path")
                if train_data_str:
                    train_data_path = _apply_run_id_to_path(
                        train_data_str,
                        run_id,
                        default_filename="preprocessed_train.pkl",
                    )
                    # Try multiple filename patterns
                    patterns = [
                        train_data_path.name.replace("train", "val"),
                        train_data_path.name.replace("train", "validation"),
                        train_data_path.name.replace("_train", "_val"),
                        "preprocessed_val.pkl",
                    ]
                    for pattern in patterns:
                        candidate = train_data_path.parent / pattern
                        if candidate not in [p[1] for p in searched_paths]:
                            searched_paths.append(
                                ("Same directory as train data", candidate)
                            )
                        if candidate.exists():
                            data = candidate
                            click.echo(f"  ✓ Found validation data: {data}")
                            break

            # Try 3: Look in preprocess output directory
            if data is None:
                preprocess_output_str = cfg.get("preprocess.output_dir")
                if preprocess_output_str:
                    preprocess_dir = Path(preprocess_output_str)
                    for filename in [
                        "preprocessed_val.pkl",
                        "val.pkl",
                        "validation.pkl",
                    ]:
                        candidate = preprocess_dir / filename
                        if candidate not in [p[1] for p in searched_paths]:
                            searched_paths.append(
                                ("Preprocess output directory", candidate)
                            )
                        if candidate.exists():
                            data = candidate
                            click.echo(f"  ✓ Found validation data: {data}")
                            break

            # Try 4: Default paths
            if data is None:
                default_paths = [
                    Path("./data/retriever")
                    / (run_id if run_id else ".")
                    / "preprocessed_val.pkl",
                    Path("./data/retriever/preprocessed_val.pkl"),
                    Path("./data/preprocessed_val.pkl"),
                ]
                for candidate in default_paths:
                    candidate = candidate.resolve()
                    if candidate not in [p[1] for p in searched_paths]:
                        searched_paths.append(("Default location", candidate))
                    if candidate.exists():
                        data = candidate
                        click.echo(f"  ✓ Found validation data: {data}")
                        break

            # Try 5: Use training data as fallback
            if data is None:
                train_data_str = cfg.get("retriever.preprocessed_data_path")
                if train_data_str:
                    train_data_path = _apply_run_id_to_path(
                        train_data_str,
                        run_id,
                        default_filename="preprocessed_train.pkl",
                    )
                    if train_data_path.exists():
                        click.echo(
                            click.style(
                                f"\n⚠ Warning: No validation data found. Using training data as fallback: {train_data_path}",
                                fg="yellow",
                            )
                        )
                        click.echo(
                            click.style(
                                "  (This is not recommended - validation should use separate data)",
                                fg="yellow",
                            )
                        )
                        data = train_data_path

            # If still not found, show error with detailed search paths
            if data is None:
                click.echo(
                    click.style(
                        "\n✗ Error: Could not find validation data.",
                        fg="red",
                        bold=True,
                    ),
                    err=True,
                )
                click.echo("\nSearched in the following locations:", err=True)
                for i, (source, path) in enumerate(searched_paths, 1):
                    click.echo(f"  {i}. {source}:", err=True)
                    click.echo(f"     {path}", err=True)

                click.echo(
                    click.style(
                        "\nTo fix this issue:",
                        fg="yellow",
                    ),
                    err=True,
                )
                click.echo(
                    "  1. Create validation data using: docs2synth retriever preprocess",
                    err=True,
                )
                click.echo(
                    "  2. Or specify the path explicitly: --data /path/to/val.pkl",
                    err=True,
                )
                click.echo(
                    "  3. Or set retriever.validation_data_path in config.yml",
                    err=True,
                )
                sys.exit(1)
        else:
            data = Path(data)

        # Auto-discover model path if not provided
        if model is None:
            click.echo(
                click.style(
                    "No model path provided, attempting auto-discovery...", fg="yellow"
                )
            )
            model = _auto_discover_model_path(cfg, run_id)
            if model is None:
                click.echo(
                    click.style(
                        "✗ Error: Could not auto-discover model. Please specify --model explicitly.",
                        fg="red",
                    ),
                    err=True,
                )
                click.echo(
                    "\nSearched in:",
                    err=True,
                )
                click.echo(
                    "  1. Config retriever.model_path",
                    err=True,
                )
                click.echo(
                    "  2. Default: ./models/retriever/final_model.pth",
                    err=True,
                )
                click.echo(
                    "  3. Latest checkpoint in retriever.checkpoint_dir",
                    err=True,
                )
                sys.exit(1)
        else:
            model = Path(model)

        # Set default output directory - organize by run_id in models/retriever folder
        if output is None:
            # Use run_id to organize validation reports with model artifacts
            if run_id:
                output = Path(f"./models/retriever/{run_id}/validation_reports")
            else:
                output = Path("./models/retriever/validation_reports")
            click.echo(f"  Using default output directory: {output}")
        else:
            output = Path(output)
        output.mkdir(parents=True, exist_ok=True)

        click.echo(
            click.style(
                "\nValidating retriever training results...", fg="blue", bold=True
            )
        )
        click.echo(f"  Model: {model}")
        click.echo(f"  Data:  {data}")
        click.echo(f"  Mode:  {mode}")
        click.echo(f"  Output: {output}\n")

        # Load model
        click.echo(click.style("Loading model...", fg="yellow"))
        try:
            trained_model = _load_model_for_validation(model)
        except Exception as e:
            click.echo(click.style(f"✗ Error loading model: {e}", fg="red"), err=True)
            logger.exception("Model loading failed")
            sys.exit(1)

        # Set up device from config or CLI override
        device_obj, device_desc = _get_device_from_config(cfg, device)
        click.echo(f"  Device: {device_desc}")

        # Move model to device and set to eval mode
        click.echo(f"  Moving model to {device_obj}...")
        trained_model = trained_model.to(device_obj)
        trained_model.eval()

        # Load validation data
        click.echo(click.style("Loading validation data...", fg="yellow"))
        try:
            val_dataloader = _load_validation_data(data)
        except Exception as e:
            click.echo(click.style(f"✗ Error loading data: {e}", fg="red"), err=True)
            logger.exception("Data loading failed")
            sys.exit(1)

        # Run evaluation
        click.echo(click.style("\nRunning evaluation...", fg="yellow"))
        try:
            # Temporarily override _get_device to use the same device as the model
            import docs2synth.retriever.training as training_module

            original_get_device = training_module._get_device
            training_module._get_device = lambda: device_obj

            try:
                if mode.lower() == "layout":
                    from docs2synth.retriever import evaluate_layout

                    eval_func = evaluate_layout
                else:
                    from docs2synth.retriever import evaluate

                    eval_func = evaluate

                results = eval_func(trained_model, val_dataloader)
            finally:
                # Restore original _get_device
                training_module._get_device = original_get_device
            (
                average_anls,
                average_loss,
                pred_texts,
                gt_texts,
                pred_entities,
                target_entities,
            ) = results

            click.echo(
                click.style(
                    f"  ✓ Evaluation complete\n    ANLS: {average_anls:.4f} | Loss: {average_loss:.4f}",
                    fg="green",
                )
            )
        except Exception as e:
            click.echo(
                click.style(f"✗ Error during evaluation: {e}", fg="red"), err=True
            )
            logger.exception("Evaluation failed")
            sys.exit(1)

        # Detailed analysis
        click.echo(click.style("\nPerforming detailed analysis...", fg="yellow"))
        try:
            validator = TrainingValidator(output_dir=output)
            analysis = validator.analyze_predictions(
                pred_texts=pred_texts,
                gt_texts=gt_texts,
                pred_entities=pred_entities,
                target_entities=target_entities,
                save_path=output / "detailed_analysis.txt",
            )
            click.echo(click.style("  ✓ Analysis complete\n", fg="green"))

            # Print results
            click.echo(click.style("=" * 70, fg="blue"))
            click.echo(click.style("VALIDATION RESULTS", fg="blue", bold=True))
            click.echo(click.style("=" * 70, fg="blue"))

            _print_validation_summary(analysis, pred_texts)

            checks = validator.sanity_check_batch(
                pred_texts=pred_texts,
                gt_texts=gt_texts,
                pred_entities=pred_entities,
                target_entities=target_entities,
            )
            _print_sanity_checks(checks, pred_texts)
            _print_worst_predictions(analysis)

            # Output files
            click.echo("\n" + click.style("=" * 70, fg="blue"))
            click.echo(click.style("OUTPUT FILES", fg="blue", bold=True))
            click.echo(click.style("=" * 70, fg="blue"))
            click.echo(f"  Detailed analysis: {output / 'detailed_analysis.txt'}")

            # Try to generate plots
            try:
                import matplotlib  # noqa: F401

                validator.history["train_anls"] = [average_anls]
                validator.history["train_entity_acc"] = [analysis["entity"]["accuracy"]]
                validator.plot_training_curves(
                    save_path=output / "validation_metrics.png"
                )
                click.echo(f"  Metrics plot:      {output / 'validation_metrics.png'}")
            except ImportError:
                click.echo(
                    click.style(
                        "  (matplotlib not installed, skipping plots)", fg="yellow"
                    )
                )

            # Final summary
            click.echo("\n" + click.style("=" * 70, fg="green", bold=True))
            click.echo(click.style("✓ VALIDATION COMPLETE", fg="green", bold=True))
            click.echo(click.style("=" * 70, fg="green", bold=True) + "\n")

        except Exception as e:
            click.echo(click.style(f"✗ Error during analysis: {e}", fg="red"), err=True)
            logger.exception("Analysis failed")
            sys.exit(1)

    except Exception as e:
        logger.exception("Retriever validation failed")
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
