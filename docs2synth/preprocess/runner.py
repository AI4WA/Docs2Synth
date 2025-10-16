"""Preprocess orchestration utilities.

This module centralizes the logic to run a chosen processor over a single file
or a directory of files and write schema-compliant outputs to disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from docs2synth.utils.config import Config
from docs2synth.utils.logging import get_logger

logger = get_logger(__name__)


def _get_processor(name: str):
    name_lower = name.lower()
    if name_lower == "paddleocr":
        from .paddleocr import PaddleOCRProcessor

        return PaddleOCRProcessor()
    raise ValueError(f"Unsupported processor: {name}")


def run_preprocess(
    path: str | Path,
    *,
    processor: str = "paddleocr",
    output_dir: Optional[str | Path] = None,
    lang: Optional[str] = None,
    device: Optional[str] = None,
    config: Optional[Config] = None,
) -> Tuple[int, int, List[Path]]:
    """Run preprocessing on a file or directory.

    Parameters
    ----------
    path : str | Path
        File or directory to process.
    processor : str
        Processor name (default: "paddleocr").
    output_dir : Optional[str | Path]
        Directory to write outputs (defaults to config.data.processed_dir).
    lang : Optional[str]
        Optional language override passed to the processor (if supported).
    config : Optional[Config]
        Configuration instance; if None, a default will be used.

    Returns
    -------
    (num_success, num_failed, output_paths)
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")

    cfg = config or Config()
    # Determine the base processed directory
    base_out_root = (
        Path(output_dir).resolve()
        if output_dir is not None
        else Path(cfg.get("data.processed_dir", "./data/processed")).resolve()
    )

    # If input is a directory, write outputs into a subfolder under processed_dir
    # named after the input directory (e.g., processed_dir/folderA)
    if p.is_dir():
        out_root = (base_out_root / p.name).resolve()
    else:
        out_root = base_out_root

    out_root.mkdir(parents=True, exist_ok=True)

    if p.is_dir():
        files = sorted([fp for fp in p.iterdir() if fp.is_file()])
    else:
        files = [p]

    proc = _get_processor(processor)

    num_success = 0
    num_failed = 0
    outputs: List[Path] = []

    for f in files:
        try:
            # Processor may accept lang override
            kwargs = {}
            if lang is not None:
                kwargs["lang"] = lang
            if device is not None:
                kwargs["device"] = device
            result = proc.process(str(f), **kwargs)  # type: ignore[arg-type]
            out_path = out_root / (f.stem + ".json")
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(result.to_json(indent=2))
            num_success += 1
            outputs.append(out_path)
            logger.info(f"Processed {f} -> {out_path}")
        except Exception as e:  # pragma: no cover - exercised via CLI typically
            num_failed += 1
            logger.exception(f"Failed processing {f}: {e}")

    return num_success, num_failed, outputs
