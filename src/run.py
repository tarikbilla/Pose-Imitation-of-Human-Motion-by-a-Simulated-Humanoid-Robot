"""Command-line entrypoint for the pose imitation pipeline."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.pipeline import PipelineOptions, PoseImitationPipeline
from src.utils.config import load_config

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pose-imitation",
        description="Markerless pose imitation pipeline for a simulated humanoid robot.",
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to YAML configuration file (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--source", default=None,
        help="Override input source: webcam index (e.g. '0') or path to a video file.",
    )
    parser.add_argument(
        "--no-webots", action="store_true",
        help="Disable Webots UDP bridge (pure perception demo).",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Run headless (no OpenCV window). Useful for SSH/servers.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Stop after N frames (0 = unlimited).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOG_FORMAT)
    log = logging.getLogger("pose_imitation")

    config_path = Path(args.config)
    if not config_path.exists():
        log.error("Config file not found: %s", config_path)
        return 2

    config = load_config(config_path)
    options = PipelineOptions(
        config=config,
        show_window=not args.no_display,
        enable_webots=not args.no_webots,
        max_frames=args.max_frames,
        source_override=args.source,
    )

    log.info("Starting pipeline (display=%s, webots=%s)",
             options.show_window, options.enable_webots)
    pipeline = PoseImitationPipeline(options=options)
    return pipeline.run()


if __name__ == "__main__":
    sys.exit(main())
