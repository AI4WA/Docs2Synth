"""Training validation and analysis utilities.

This module provides tools to validate training results and ensure correctness:
- Sanity checks during training
- Detailed metrics analysis
- Training curve visualization
- Error analysis
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from docs2synth.retriever.metrics import calculate_anls
from docs2synth.utils.logging import get_logger

logger = get_logger(__name__)


class TrainingValidator:
    """Validates training results and provides sanity checks."""

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize validator.

        Args:
            output_dir: Directory to save validation reports and plots
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_anls": [],
            "train_entity_acc": [],
            "val_loss": [],
            "val_anls": [],
            "val_entity_acc": [],
        }

    def sanity_check_batch(
        self,
        pred_texts: List[str],
        gt_texts: List[str],
        pred_entities: List[int],
        target_entities: List[int],
        batch_idx: int = 0,
    ) -> Dict[str, Any]:
        """Perform sanity checks on a training batch.

        Args:
            pred_texts: Predicted answer texts
            gt_texts: Ground truth answer texts
            pred_entities: Predicted entity IDs
            target_entities: Target entity IDs
            batch_idx: Batch index for logging

        Returns:
            Dictionary with sanity check results
        """
        checks = {}

        # 1. Check for empty predictions (model not learning)
        empty_preds = sum(1 for p in pred_texts if not p.strip())
        checks["empty_predictions"] = empty_preds
        if empty_preds > len(pred_texts) * 0.5:
            logger.warning(
                f"Batch {batch_idx}: {empty_preds}/{len(pred_texts)} predictions are empty! "
                "Model may not be learning properly."
            )

        # 2. Check for identical predictions (model collapsed)
        unique_preds = len(set(pred_texts))
        checks["unique_predictions"] = unique_preds
        if unique_preds < len(pred_texts) * 0.1 and len(pred_texts) > 10:
            logger.warning(
                f"Batch {batch_idx}: Only {unique_preds} unique predictions out of {len(pred_texts)}! "
                "Model may have collapsed."
            )

        # 3. Check entity prediction diversity
        unique_entities = len(set(pred_entities))
        checks["unique_entities"] = unique_entities
        if unique_entities == 1 and len(pred_entities) > 10:
            logger.warning(
                f"Batch {batch_idx}: All entity predictions are {pred_entities[0]}! "
                "Entity retrieval head may not be learning."
            )

        # 4. Calculate entity accuracy
        entity_acc = sum(
            1 for p, t in zip(pred_entities, target_entities) if p == t
        ) / len(pred_entities)
        checks["entity_accuracy"] = entity_acc

        # 5. Calculate average ANLS
        anls_scores = [calculate_anls(p, g) for p, g in zip(pred_texts, gt_texts)]
        checks["average_anls"] = np.mean(anls_scores)

        return checks

    def check_training_progress(
        self,
        epoch: int,
        train_loss: float,
        train_anls: float,
        train_entity_acc: float,
        val_loss: Optional[float] = None,
        val_anls: Optional[float] = None,
        val_entity_acc: Optional[float] = None,
    ) -> Dict[str, str]:
        """Check if training is progressing correctly.

        Args:
            epoch: Current epoch number
            train_loss: Training loss
            train_anls: Training ANLS score
            train_entity_acc: Training entity accuracy
            val_loss: Validation loss (optional)
            val_anls: Validation ANLS score (optional)
            val_entity_acc: Validation entity accuracy (optional)

        Returns:
            Dictionary with warning messages (empty if no issues)
        """
        warnings = {}

        # Update history
        self.history["train_loss"].append(train_loss)
        self.history["train_anls"].append(train_anls)
        self.history["train_entity_acc"].append(train_entity_acc)
        if val_loss is not None:
            self.history["val_loss"].append(val_loss)
            self.history["val_anls"].append(val_anls or 0.0)
            self.history["val_entity_acc"].append(val_entity_acc or 0.0)

        # Skip checks for first epoch
        if epoch < 1:
            return warnings

        # 1. Check if loss is decreasing
        if len(self.history["train_loss"]) >= 3:
            recent_losses = self.history["train_loss"][-3:]
            if all(recent_losses[i] >= recent_losses[i - 1] for i in range(1, 3)):
                warnings["loss_not_decreasing"] = (
                    f"Training loss has not decreased for 3 epochs: {recent_losses}. "
                    "Consider lowering learning rate or checking data quality."
                )

        # 2. Check if ANLS is improving
        if len(self.history["train_anls"]) >= 3:
            recent_anls = self.history["train_anls"][-3:]
            if all(recent_anls[i] <= recent_anls[i - 1] for i in range(1, 3)):
                warnings["anls_not_improving"] = (
                    f"Training ANLS has not improved for 3 epochs: {recent_anls}. "
                    "Model may have reached plateau."
                )

        # 3. Check for overfitting
        if val_anls is not None and len(self.history["val_anls"]) >= 2:
            val_drop = train_anls - val_anls
            if val_drop > 0.2:
                warnings["overfitting"] = (
                    f"Large gap between train ANLS ({train_anls:.4f}) "
                    f"and val ANLS ({val_anls:.4f}). Model may be overfitting."
                )

        # 4. Check if metrics are too low (data quality issue)
        if epoch >= 5 and train_anls < 0.1:
            warnings["low_anls"] = (
                f"Training ANLS is very low ({train_anls:.4f}) after {epoch} epochs. "
                "Check data quality and preprocessing."
            )

        # 5. Check for NaN/inf loss
        if np.isnan(train_loss) or np.isinf(train_loss):
            warnings["invalid_loss"] = (
                f"Training loss is {train_loss}! Check learning rate and gradients."
            )

        return warnings

    def analyze_predictions(
        self,
        pred_texts: List[str],
        gt_texts: List[str],
        pred_entities: List[int],
        target_entities: List[int],
        save_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Analyze predictions in detail.

        Args:
            pred_texts: All predicted answer texts
            gt_texts: All ground truth answer texts
            pred_entities: All predicted entity IDs
            target_entities: All target entity IDs
            save_path: Path to save detailed analysis

        Returns:
            Dictionary with analysis results
        """
        analysis = {}

        # 1. ANLS score distribution
        anls_scores = [calculate_anls(p, g) for p, g in zip(pred_texts, gt_texts)]
        analysis["anls"] = {
            "mean": np.mean(anls_scores),
            "std": np.std(anls_scores),
            "median": np.median(anls_scores),
            "min": np.min(anls_scores),
            "max": np.max(anls_scores),
            "perfect_matches": sum(1 for s in anls_scores if s == 1.0),
            "zero_matches": sum(1 for s in anls_scores if s == 0.0),
        }

        # 2. Entity accuracy
        entity_correct = sum(
            1 for p, t in zip(pred_entities, target_entities) if p == t
        )
        analysis["entity"] = {
            "accuracy": entity_correct / len(pred_entities),
            "total": len(pred_entities),
            "correct": entity_correct,
        }

        # 3. Error analysis - find worst predictions
        errors = [
            {
                "predicted": p,
                "ground_truth": g,
                "anls": s,
                "entity_pred": pe,
                "entity_target": te,
            }
            for p, g, s, pe, te in zip(
                pred_texts, gt_texts, anls_scores, pred_entities, target_entities
            )
        ]
        errors_sorted = sorted(errors, key=lambda x: x["anls"])
        analysis["worst_predictions"] = errors_sorted[:10]
        analysis["best_predictions"] = errors_sorted[-10:]

        # 4. Prediction length statistics
        pred_lengths = [len(p.split()) for p in pred_texts]
        gt_lengths = [len(g.split()) for g in gt_texts]
        analysis["length"] = {
            "pred_mean": np.mean(pred_lengths),
            "gt_mean": np.mean(gt_lengths),
            "pred_std": np.std(pred_lengths),
            "gt_std": np.std(gt_lengths),
        }

        # 5. Save detailed report
        if save_path:
            self._save_analysis_report(analysis, save_path)

        return analysis

    def _save_analysis_report(self, analysis: Dict[str, Any], path: Path) -> None:
        """Save detailed analysis report to file."""
        with open(path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("TRAINING VALIDATION REPORT\n")
            f.write("=" * 70 + "\n\n")

            # ANLS Analysis
            f.write("ANLS Score Analysis:\n")
            f.write("-" * 50 + "\n")
            anls = analysis["anls"]
            f.write(f"  Mean:            {anls['mean']:.4f}\n")
            f.write(f"  Std:             {anls['std']:.4f}\n")
            f.write(f"  Median:          {anls['median']:.4f}\n")
            f.write(f"  Min:             {anls['min']:.4f}\n")
            f.write(f"  Max:             {anls['max']:.4f}\n")
            f.write(f"  Perfect matches: {anls['perfect_matches']}\n")
            f.write(f"  Zero matches:    {anls['zero_matches']}\n\n")

            # Entity Analysis
            f.write("Entity Retrieval Analysis:\n")
            f.write("-" * 50 + "\n")
            entity = analysis["entity"]
            f.write(f"  Accuracy: {entity['accuracy']:.4f}\n")
            f.write(f"  Correct:  {entity['correct']} / {entity['total']}\n\n")

            # Length Analysis
            f.write("Prediction Length Analysis:\n")
            f.write("-" * 50 + "\n")
            length = analysis["length"]
            f.write(f"  Predicted mean: {length['pred_mean']:.2f} words\n")
            f.write(f"  Ground truth mean: {length['gt_mean']:.2f} words\n\n")

            # Worst predictions
            f.write("Top 10 Worst Predictions:\n")
            f.write("-" * 50 + "\n")
            for i, error in enumerate(analysis["worst_predictions"], 1):
                f.write(f"\n{i}. ANLS: {error['anls']:.4f}\n")
                f.write(f"   Predicted:     {error['predicted']}\n")
                f.write(f"   Ground Truth:  {error['ground_truth']}\n")
                f.write(
                    f"   Entity: {error['entity_pred']} (target: {error['entity_target']})\n"
                )

            f.write("\n" + "=" * 70 + "\n")

        logger.info(f"Detailed analysis saved to {path}")

    def plot_training_curves(self, save_path: Optional[Path] = None) -> None:
        """Plot training curves (loss and ANLS).

        Args:
            save_path: Path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed, skipping plot generation")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Training Loss
        axes[0, 0].plot(self.history["train_loss"], label="Train Loss", marker="o")
        if self.history["val_loss"]:
            axes[0, 0].plot(self.history["val_loss"], label="Val Loss", marker="s")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Plot 2: ANLS Score
        axes[0, 1].plot(self.history["train_anls"], label="Train ANLS", marker="o")
        if self.history["val_anls"]:
            axes[0, 1].plot(self.history["val_anls"], label="Val ANLS", marker="s")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("ANLS Score")
        axes[0, 1].set_title("ANLS Score")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Plot 3: Entity Accuracy
        axes[1, 0].plot(
            self.history["train_entity_acc"], label="Train Entity Acc", marker="o"
        )
        if self.history["val_entity_acc"]:
            axes[1, 0].plot(
                self.history["val_entity_acc"], label="Val Entity Acc", marker="s"
            )
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].set_title("Entity Retrieval Accuracy")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Plot 4: Learning Progress (combined metric)
        combined_train = [
            a * e
            for a, e in zip(
                self.history["train_anls"], self.history["train_entity_acc"]
            )
        ]
        axes[1, 1].plot(combined_train, label="Train (ANLS × Entity)", marker="o")
        if self.history["val_anls"] and self.history["val_entity_acc"]:
            combined_val = [
                a * e
                for a, e in zip(
                    self.history["val_anls"], self.history["val_entity_acc"]
                )
            ]
            axes[1, 1].plot(combined_val, label="Val (ANLS × Entity)", marker="s")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Combined Score")
        axes[1, 1].set_title("Combined Performance (ANLS × Entity Accuracy)")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Training curves saved to {save_path}")
        else:
            plt.show()

        plt.close()


def calculate_entity_accuracy(
    pred_entities: List[int], target_entities: List[int]
) -> float:
    """Calculate entity retrieval accuracy.

    Args:
        pred_entities: Predicted entity IDs
        target_entities: Target entity IDs

    Returns:
        Accuracy score between 0.0 and 1.0
    """
    if not pred_entities:
        return 0.0
    correct = sum(1 for p, t in zip(pred_entities, target_entities) if p == t)
    return correct / len(pred_entities)
