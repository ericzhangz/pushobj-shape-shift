"""Hydra-compatible entrypoint for baseline capture and pilot instrumentation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--pilot-mode",
        choices=("capture", "off", "on"),
        required=True,
        help="capture/off only save returned actions; on enables diagnostics",
    )
    parser.add_argument("--pilot-output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument(
        "--oracle-steps",
        default="0,24,49,74,99",
        help="Pre-registered zero-based GD iterations for environment branches",
    )
    return parser.parse_known_args()


def main() -> None:
    args, hydra_args = _parse_args()
    oracle_steps = tuple(
        int(value.strip()) for value in args.oracle_steps.split(",") if value.strip()
    )

    # Import plan before installing wrappers so Hydra uses the patched class objects.
    import plan
    from research.instrumentation import PilotRecorder, install_instrumentation

    recorder = PilotRecorder(
        output_dir=args.pilot_output,
        run_id=args.run_id,
        arm=args.arm,
        enabled=args.pilot_mode == "on",
        oracle_steps=oracle_steps,
    )
    install_instrumentation(recorder)
    sys.argv = [sys.argv[0], *hydra_args]
    plan.main()


if __name__ == "__main__":
    main()

