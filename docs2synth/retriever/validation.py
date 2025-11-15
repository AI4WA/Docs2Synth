"""Training validation and analysis utilities.

This module provides tools to validate training results and ensure correctness:
- Sanity checks during training
- Detailed metrics analysis
- Training curve visualization
- Error analysis
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

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
        batch_idx: int = 0,
    ) -> Dict[str, Any]:
        """Perform sanity checks on a training batch.

        Args:
            pred_texts: Predicted answer texts
            gt_texts: Ground truth answer texts
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

        # 3. Calculate average ANLS
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
        save_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Analyze predictions in detail.

        Args:
            pred_texts: All predicted answer texts
            gt_texts: All ground truth answer texts
            save_path: Path to save detailed analysis

        Returns:
            Dictionary with analysis results
        """
        # Check for empty inputs
        if not pred_texts or not gt_texts:
            logger.warning(
                "Empty predictions or ground truth provided, returning empty analysis"
            )
            return {
                "anls": {
                    "mean": 0.0,
                    "std": 0.0,
                    "median": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "perfect_matches": 0,
                    "zero_matches": 0,
                    "quartiles": {"q25": 0.0, "q75": 0.0},
                },
                "error_patterns": {
                    "empty_predictions": len(pred_texts) if pred_texts else 0,
                    "empty_percentage": 100.0 if not pred_texts else 0.0,
                    "partial_matches": 0,
                    "partial_percentage": 0.0,
                },
                "by_length": {
                    "short": {"count": 0, "mean_anls": 0.0},
                    "medium": {"count": 0, "mean_anls": 0.0},
                    "long": {"count": 0, "mean_anls": 0.0},
                },
                "worst_predictions": [],
                "best_predictions": [],
                "length": {
                    "pred_mean": 0.0,
                    "gt_mean": 0.0,
                    "pred_std": 0.0,
                    "gt_std": 0.0,
                    "pred_max": 0,
                    "gt_max": 0,
                },
            }

        analysis = {}

        # 1. ANLS score distribution
        anls_scores = [calculate_anls(p, g) for p, g in zip(pred_texts, gt_texts)]

        # Additional check: ensure anls_scores is not empty after calculation
        if not anls_scores:
            logger.warning("No ANLS scores calculated, returning empty analysis")
            return {
                "anls": {
                    "mean": 0.0,
                    "std": 0.0,
                    "median": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "perfect_matches": 0,
                    "zero_matches": 0,
                    "quartiles": {"q25": 0.0, "q75": 0.0},
                },
                "error_patterns": {
                    "empty_predictions": len(pred_texts),
                    "empty_percentage": 100.0,
                    "partial_matches": 0,
                    "partial_percentage": 0.0,
                },
                "by_length": {
                    "short": {"count": 0, "mean_anls": 0.0},
                    "medium": {"count": 0, "mean_anls": 0.0},
                    "long": {"count": 0, "mean_anls": 0.0},
                },
                "worst_predictions": [],
                "best_predictions": [],
                "length": {
                    "pred_mean": 0.0,
                    "gt_mean": 0.0,
                    "pred_std": 0.0,
                    "gt_std": 0.0,
                    "pred_max": 0,
                    "gt_max": 0,
                },
            }
        analysis["anls"] = {
            "mean": np.mean(anls_scores),
            "std": np.std(anls_scores),
            "median": np.median(anls_scores),
            "min": np.min(anls_scores),
            "max": np.max(anls_scores),
            "perfect_matches": sum(1 for s in anls_scores if s == 1.0),
            "zero_matches": sum(1 for s in anls_scores if s == 0.0),
            "quartiles": {
                "q25": np.percentile(anls_scores, 25),
                "q75": np.percentile(anls_scores, 75),
            },
        }

        # 2. Error pattern analysis
        empty_preds = sum(1 for p in pred_texts if not p.strip())
        partial_matches = sum(1 for s in anls_scores if 0 < s < 1.0)

        analysis["error_patterns"] = {
            "empty_predictions": empty_preds,
            "empty_percentage": (
                (empty_preds / len(pred_texts) * 100) if pred_texts else 0.0
            ),
            "partial_matches": partial_matches,
            "partial_percentage": (
                (partial_matches / len(anls_scores) * 100) if anls_scores else 0.0
            ),
        }

        # 3. Error analysis by answer length
        length_categories = {"short": [], "medium": [], "long": []}
        for p, g, s in zip(pred_texts, gt_texts, anls_scores):
            gt_len = len(g.split())
            if gt_len <= 2:
                length_categories["short"].append(s)
            elif gt_len <= 5:
                length_categories["medium"].append(s)
            else:
                length_categories["long"].append(s)

        analysis["by_length"] = {
            cat: {
                "count": len(scores),
                "mean_anls": np.mean(scores) if scores else 0.0,
            }
            for cat, scores in length_categories.items()
        }

        # 4. Error analysis - find worst predictions
        errors = [
            {
                "predicted": p,
                "ground_truth": g,
                "anls": s,
                "pred_len": len(p.split()),
                "gt_len": len(g.split()),
                "is_empty": not p.strip(),
            }
            for p, g, s in zip(pred_texts, gt_texts, anls_scores)
        ]
        errors_sorted = sorted(errors, key=lambda x: x["anls"])
        analysis["worst_predictions"] = errors_sorted[:10]
        analysis["best_predictions"] = errors_sorted[-10:]

        # 6. Prediction length statistics
        pred_lengths = [len(p.split()) for p in pred_texts]
        gt_lengths = [len(g.split()) for g in gt_texts]
        analysis["length"] = {
            "pred_mean": np.mean(pred_lengths) if pred_lengths else 0.0,
            "gt_mean": np.mean(gt_lengths) if gt_lengths else 0.0,
            "pred_std": np.std(pred_lengths) if pred_lengths else 0.0,
            "gt_std": np.std(gt_lengths) if gt_lengths else 0.0,
            "pred_max": np.max(pred_lengths) if pred_lengths else 0,
            "gt_max": np.max(gt_lengths) if gt_lengths else 0,
        }

        # 7. Save detailed report
        if save_path:
            self._save_analysis_report(analysis, save_path)

        return analysis

    def _write_anls_section(self, f, anls: Dict[str, Any]) -> None:
        """Write ANLS score analysis section."""
        f.write("ANLS Score Analysis:\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Mean:            {anls['mean']:.4f}\n")
        f.write(f"  Std:             {anls['std']:.4f}\n")
        f.write(f"  Median:          {anls['median']:.4f}\n")
        f.write(f"  Min:             {anls['min']:.4f}\n")
        f.write(f"  Max:             {anls['max']:.4f}\n")
        f.write(f"  Perfect matches: {anls['perfect_matches']}\n")
        f.write(f"  Zero matches:    {anls['zero_matches']}\n")
        f.write(f"  Q1 (25th):       {anls['quartiles']['q25']:.4f}\n")
        f.write(f"  Q3 (75th):       {anls['quartiles']['q75']:.4f}\n\n")

    def _write_error_patterns_section(self, f, errors: Dict[str, Any]) -> None:
        """Write error pattern analysis section."""
        f.write("Error Pattern Analysis:\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Empty predictions:    {errors['empty_predictions']} ")
        f.write(f"({errors['empty_percentage']:.1f}%)\n")
        f.write(f"  Partial matches:      {errors['partial_matches']} ")
        f.write(f"({errors['partial_percentage']:.1f}%)\n\n")

        if errors["empty_percentage"] > 50:
            f.write("  ⚠️  CRITICAL: >50% predictions are empty!\n")
            f.write("      → Model is not generating outputs properly\n")
            f.write("      → Check: decoder configuration, generation parameters\n")
            f.write("      → Verify: training loss is decreasing\n\n")
        elif errors["empty_percentage"] > 20:
            f.write("  ⚠️  WARNING: >20% predictions are empty\n")
            f.write("      → Model may need more training epochs\n")
            f.write("      → Consider: adjusting learning rate or batch size\n\n")

    def _write_length_performance_section(self, f, by_length: Dict[str, Any]) -> None:
        """Write performance by answer length section."""
        f.write("Performance by Answer Length:\n")
        f.write("-" * 50 + "\n")
        for cat in ["short", "medium", "long"]:
            cat_data = by_length[cat]
            f.write(f"  {cat.capitalize()} answers:  ")
            f.write(f"ANLS={cat_data['mean_anls']:.4f}, ")
            f.write(f"count={cat_data['count']}\n")

        f.write("\n  (short ≤2 words, medium ≤5 words, long >5 words)\n")

        if by_length["short"]["count"] > 0 and by_length["long"]["count"] > 0:
            short_anls = by_length["short"]["mean_anls"]
            long_anls = by_length["long"]["mean_anls"]
            if short_anls > long_anls + 0.2:
                f.write("\n  💡 Model performs better on short answers\n")
                f.write("     → Consider: training more on long-form answers\n")
            elif long_anls > short_anls + 0.2:
                f.write("\n  💡 Model performs better on long answers\n")
        f.write("\n")

    def _write_length_statistics_section(self, f, length: Dict[str, Any]) -> None:
        """Write prediction length statistics section."""
        f.write("Prediction Length Statistics:\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Predicted mean:    {length['pred_mean']:.2f} words ")
        f.write(f"(max: {length['pred_max']})\n")
        f.write(f"  Ground truth mean: {length['gt_mean']:.2f} words ")
        f.write(f"(max: {length['gt_max']})\n")
        f.write(f"  Predicted std:     {length['pred_std']:.2f}\n")
        f.write(f"  Ground truth std:  {length['gt_std']:.2f}\n")

        if length["pred_mean"] < length["gt_mean"] * 0.5:
            f.write(
                "\n  ⚠️  WARNING: Predictions are significantly shorter than ground truth\n"
            )
            f.write("      → Model may be undertrained or truncating outputs\n")
        elif length["pred_mean"] > length["gt_mean"] * 1.5:
            f.write(
                "\n  ⚠️  WARNING: Predictions are significantly longer than ground truth\n"
            )
            f.write("      → Model may be generating excessive text\n")
        f.write("\n")

    def _write_predictions_section(self, f, analysis: Dict[str, Any]) -> None:
        """Write worst and best predictions sections."""
        f.write("Top 10 Worst Predictions:\n")
        f.write("-" * 50 + "\n")
        for i, error in enumerate(analysis["worst_predictions"], 1):
            f.write(f"\n{i}. ANLS: {error['anls']:.4f}")
            if error["is_empty"]:
                f.write(" [EMPTY]")
            f.write("\n")
            f.write(f"   Predicted ({error['pred_len']} words):  ")
            f.write(f"{error['predicted'] if error['predicted'] else '(empty)'}\n")
            f.write(f"   Ground Truth ({error['gt_len']} words): ")
            f.write(f"{error['ground_truth']}\n")

        f.write("\n\nTop 10 Best Predictions (for reference):\n")
        f.write("-" * 50 + "\n")
        for i, success in enumerate(analysis["best_predictions"], 1):
            f.write(f"\n{i}. ANLS: {success['anls']:.4f}\n")
            f.write(f"   Predicted ({success['pred_len']} words):  ")
            f.write(f"{success['predicted'] if success['predicted'] else '(empty)'}\n")
            f.write(f"   Ground Truth ({success['gt_len']} words): ")
            f.write(f"{success['ground_truth']}\n")

    def _write_assessment_section(
        self, f, mean_anls: float, errors: Dict[str, Any], length: Dict[str, Any]
    ) -> None:
        """Write overall assessment and recommendations section."""
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("OVERALL ASSESSMENT & RECOMMENDATIONS\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Overall ANLS Score: {mean_anls:.4f}\n\n")

        if mean_anls >= 0.9:
            f.write("✓ EXCELLENT: Model is performing very well!\n")
        elif mean_anls >= 0.7:
            f.write("✓ GOOD: Model shows solid performance\n")
        elif mean_anls >= 0.5:
            f.write("⚠️  FAIR: Model needs improvement\n")
        else:
            f.write("✗ POOR: Model requires significant improvement\n")

        f.write("\nKey Issues:\n")
        issues_found = False

        if errors["empty_percentage"] > 20:
            f.write(f"  • {errors['empty_percentage']:.1f}% empty predictions - ")
            f.write("model not generating outputs\n")
            issues_found = True

        if length["pred_mean"] < 0.1 and length["gt_mean"] > 0.3:
            f.write("  • Predictions are nearly empty - generation problem\n")
            issues_found = True

        if mean_anls < 0.5:
            f.write(f"  • Low ANLS ({mean_anls:.4f}) - answer quality is poor\n")
            issues_found = True

        if not issues_found:
            f.write("  • No critical issues detected\n")

        f.write("\nRecommendations:\n")
        if errors["empty_percentage"] > 50:
            f.write("  1. Check decoder configuration and max_length settings\n")
            f.write("  2. Verify training loss is decreasing each epoch\n")
            f.write("  3. Inspect model outputs during training (sanity checks)\n")
            f.write("  4. Consider increasing learning rate or training longer\n")
        elif errors["empty_percentage"] > 20:
            f.write("  1. Train for more epochs\n")
            f.write("  2. Review training curves for convergence\n")
        elif mean_anls < 0.7:
            f.write("  1. Increase training data if available\n")
            f.write("  2. Try different learning rates or batch sizes\n")
            f.write("  3. Consider data augmentation strategies\n")
        else:
            f.write("  1. Model is performing well - consider fine-tuning\n")
            f.write("  2. Test on additional validation sets\n")
            f.write("  3. Monitor for overfitting\n")

    def _save_analysis_report(self, analysis: Dict[str, Any], path: Path) -> None:
        """Save detailed analysis report to file."""
        with open(path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("TRAINING VALIDATION REPORT\n")
            f.write("=" * 70 + "\n\n")

            self._write_anls_section(f, analysis["anls"])
            self._write_error_patterns_section(f, analysis["error_patterns"])
            self._write_length_performance_section(f, analysis["by_length"])
            self._write_length_statistics_section(f, analysis["length"])
            self._write_predictions_section(f, analysis)
            self._write_assessment_section(
                f,
                analysis["anls"]["mean"],
                analysis["error_patterns"],
                analysis["length"],
            )

            f.write("\n" + "=" * 70 + "\n")

        logger.info(f"Detailed analysis saved to {path}")

    def save_history(self, path: Path) -> None:
        """Save training history to JSON file.

        Args:
            path: Path to save the history JSON file
        """
        import json

        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved to {path}")

    def load_history(self, path: Path) -> bool:
        """Load training history from JSON file.

        Args:
            path: Path to the history JSON file

        Returns:
            True if loaded successfully, False otherwise
        """
        import json

        if not path.exists():
            logger.warning(f"Training history file not found: {path}")
            return False

        try:
            with open(path, "r") as f:
                loaded_history = json.load(f)
            # Update history with loaded data
            for key in self.history:
                if key in loaded_history:
                    self.history[key] = loaded_history[key]
            logger.info(f"Training history loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load training history: {e}")
            return False

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

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Training Loss
        axes[0].plot(self.history["train_loss"], label="Train Loss", marker="o")
        if self.history["val_loss"]:
            axes[0].plot(self.history["val_loss"], label="Val Loss", marker="s")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].legend()
        axes[0].grid(True)

        # Plot 2: ANLS Score
        axes[1].plot(self.history["train_anls"], label="Train ANLS", marker="o")
        if self.history["val_anls"]:
            axes[1].plot(self.history["val_anls"], label="Val ANLS", marker="s")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("ANLS Score")
        axes[1].set_title("ANLS Score")
        axes[1].legend()
        axes[1].grid(True)

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
