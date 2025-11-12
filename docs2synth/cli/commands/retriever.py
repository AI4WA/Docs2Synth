"""Retriever training command-line interface commands.

This module provides CLI commands for training retriever models using
various training configurations (standard, layout, pretraining, etc.).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

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
    help="Path to training data (directory or DataLoader pickle). "
    "If not provided, reads from config.yml (retriever.train_data_path or data.processed_dir)",
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
    default=1e-5,
    help="Learning rate (default: 1e-5)",
)
@click.option(
    "--epochs",
    type=int,
    default=1,
    help="Number of training epochs (default: 1)",
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
    help="Save checkpoint every N epochs (default: save only at the end)",
)
@click.option(
    "--resume",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to checkpoint to resume training from",
)
@click.pass_context
def retriever_train(
    ctx: click.Context,
    model_path: Optional[Path],
    base_model: str,
    data_path: Path,
    output_dir: Optional[Path],
    mode: str,
    lr: float,
    epochs: int,
    batch_size: Optional[int],
    save_every: Optional[int],
    resume: Optional[Path],
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
        # Start training from HuggingFace base model (data path from config.yml)
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
        from torch.utils.data import DataLoader
    except ImportError:
        click.echo(
            click.style(
                "✗ Error: PyTorch is required for retriever training. "
                "Please install torch: pip install torch",
                fg="red",
            ),
            err=True,
        )
        sys.exit(1)

    try:
        cfg = ctx.obj.get("config")
        
        # Resolve data path from config if not provided
        if data_path is None:
            # Try retriever.train_data_path first, then fall back to data.processed_dir
            data_path_str = cfg.get("retriever.train_data_path") or cfg.get(
                "data.processed_dir"
            )
            if data_path_str:
                data_path = Path(data_path_str)
            else:
                click.echo(
                    click.style(
                        "✗ Error: --data-path is required, or set "
                        "retriever.train_data_path or data.processed_dir in config.yml",
                        fg="red",
                    ),
                    err=True,
                )
                sys.exit(1)
        else:
            data_path = Path(data_path)

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
            output_dir = Path(cfg.get("data.models_dir", "./models"))
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
            if resume:
                # Resume from checkpoint takes priority
                click.echo(f"  Resuming from checkpoint: {resume}")
                checkpoint = torch.load(resume, map_location="cpu")
                if "model_state_dict" in checkpoint:
                    # Checkpoint contains state dict, need base model to load it
                    if model_path:
                        if model_path.is_file():
                            model = torch.load(model_path, map_location="cpu")
                        elif model_path.is_dir():
                            from transformers import AutoModel

                            model = AutoModel.from_pretrained(str(model_path))
                        else:
                            raise FileNotFoundError(f"Model path does not exist: {model_path}")
                    else:
                        # Load base model and load state dict
                        from transformers import AutoModel

                        click.echo(f"  Loading base model: {base_model}")
                        model = AutoModel.from_pretrained(base_model)
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    # Checkpoint is the model itself
                    model = checkpoint
            elif model_path:
                # Load from model path
                if model_path.is_file():
                    model = torch.load(model_path, map_location="cpu")
                elif model_path.is_dir():
                    # Try loading from directory (e.g., HuggingFace format)
                    from transformers import AutoModel

                    model = AutoModel.from_pretrained(str(model_path))
                else:
                    click.echo(
                        click.style(
                            f"✗ Error: Model path does not exist: {model_path}",
                            fg="red",
                        ),
                        err=True,
                    )
                    sys.exit(1)
            else:
                # Load from HuggingFace base model
                from transformers import AutoModel

                click.echo(f"  Loading base model from HuggingFace: {base_model}")
                model = AutoModel.from_pretrained(base_model)
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

        # Load or create DataLoader
        click.echo(click.style("Loading training data...", fg="yellow"))
        try:
            if data_path.is_file() and (
                data_path.suffix == ".pkl" or data_path.suffix == ".pickle"
            ):
                # Load pickled DataLoader
                import pickle

                click.echo(f"  Loading pickled DataLoader from: {data_path}")
                with open(data_path, "rb") as f:
                    train_dataloader = pickle.load(f)
            elif data_path.is_dir():
                # Load verified QA pairs from JSON files in directory
                from docs2synth.retriever.dataset import (
                    create_dataloader_from_verified_qa,
                    load_verified_qa_pairs,
                )

                # Get processor name from config
                processor_name = cfg.get("preprocess.processor")
                if processor_name:
                    click.echo(
                        f"  Loading verified QA pairs from directory: {data_path} "
                        f"(processor: {processor_name})"
                    )
                else:
                    click.echo(
                        f"  Loading verified QA pairs from directory: {data_path} "
                        "(all processors)"
                    )

                verified_qa_pairs = load_verified_qa_pairs(
                    data_path,
                    processor_name=processor_name,
                    require_all_verifiers=True,
                )

                if not verified_qa_pairs:
                    click.echo(
                        click.style(
                            "✗ Error: No verified QA pairs found. "
                            "Please ensure JSON files contain QA pairs with verification results.",
                            fg="red",
                        ),
                        err=True,
                    )
                    sys.exit(1)

                click.echo(
                    f"  Found {len(verified_qa_pairs)} verified QA pairs, creating DataLoader..."
                )

                # Create DataLoader
                effective_batch_size = batch_size or 8
                train_dataloader = create_dataloader_from_verified_qa(
                    verified_qa_pairs,
                    batch_size=effective_batch_size,
                    shuffle=True,
                )
                click.echo(f"  Created DataLoader with batch_size={effective_batch_size}")
            else:
                click.echo(
                    click.style(
                        f"✗ Error: Unsupported data format: {data_path.suffix}. "
                        "Expected directory or .pkl/.pickle file",
                        fg="red",
                    ),
                    err=True,
                )
                sys.exit(1)
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

        # Override batch size if specified
        if batch_size is not None:
            train_dataloader.batch_size = batch_size

        # Get start epoch from checkpoint if resuming
        start_epoch = 0
        if resume:
            try:
                checkpoint = torch.load(resume, map_location="cpu")
                start_epoch = checkpoint.get("epoch", 0)
                if "optimizer_state_dict" in checkpoint:
                    logger.info("Note: Optimizer state found but not restored (new optimizer created)")
            except Exception as e:
                logger.warning(f"Could not extract epoch from checkpoint: {e}")

        # Select training function based on mode
        mode_lower = mode.lower()
        if mode_lower == "standard":
            train_func = train
        elif mode_lower == "layout":
            train_func = train_layout
        elif mode_lower == "layout-gemini":
            train_func = train_layout_gemini
        elif mode_lower == "layout-coarse-grained":
            train_func = train_layout_coarse_grained
        elif mode_lower == "pretrain-layout":
            train_func = pretrain_layout
        else:
            click.echo(
                click.style(f"✗ Error: Unknown training mode: {mode}", fg="red"),
                err=True,
            )
            sys.exit(1)

        # Move model to device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        click.echo(f"  Device: {device}")

        # Training loop
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
                average_anls, pred_texts, gt_texts, pred_entities, target_entities = (
                    train_func(model, train_dataloader, lr)
                )
                click.echo(
                    click.style(
                        f"  Average ANLS: {average_anls:.4f}",
                        fg="green",
                    )
                )
                best_anls = max(best_anls, average_anls)

                # Save checkpoint
                if save_every is None or (epoch + 1) % save_every == 0:
                    checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pth"
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state_dict": model.state_dict(),
                            "average_anls": average_anls,
                        },
                        checkpoint_path,
                    )
                    click.echo(f"  Saved checkpoint: {checkpoint_path}")

        # Save final model
        final_model_path = output_dir / "final_model.pth"
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
                f"\n✓ Training complete!",
                fg="green",
                bold=True,
            )
        )
        click.echo(f"  Final model saved: {final_model_path}")
        if mode_lower != "pretrain-layout":
            click.echo(f"  Best ANLS: {best_anls:.4f}")

    except Exception as e:
        logger.exception("Retriever training failed")
        click.echo(click.style(f"✗ Error: {e}", fg="red"), err=True)
        sys.exit(1)

