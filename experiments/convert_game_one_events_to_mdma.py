#!/usr/bin/env python3
"""Batch-convert PTZ game_one events into per-event JSON folders for MDMA."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CATEGORIES = [
    "ball_actions",
    "ball_claiming",
    "cautions",
    "crosses",
    "fouls",
    "offsides",
    "other_events",
    "passes",
    "set_pieces",
    "shots",
    "substitutions",
    "tackles",
]


@dataclass
class EventSelection:
    category: str
    source_event_dir: Path
    event_name: str
    event_index: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PTZ game_one events into MDMA experiments/game_one/events")
    parser.add_argument("--ptz-root", type=Path, required=True, help="Path to PTZ-Videography repository root")
    parser.add_argument("--mdma-root", type=Path, required=True, help="Path to MultiDroneMultiActorFilming repository root")
    parser.add_argument(
        "--max-events",
        type=int,
        default=10,
        help="Maximum number of events per category (default: 10)",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=5,
        help="Frame stride passed to converter (default: 5)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination JSON files if they already exist",
    )
    parser.add_argument(
        "--num-robots",
        type=int,
        default=5,
        help="Number of robots/cameras to encode in output JSON (default: 5)",
    )
    parser.add_argument(
        "--robot-fov",
        type=float,
        default=1.1334584350470127,
        help="Per-robot FOV value in radians (default matches existing MDMA event files)",
    )
    parser.add_argument(
        "--sense-dist",
        type=float,
        default=517.2999267578125,
        help="Sense distance value for MDMA camera model compatibility",
    )
    return parser.parse_args()


def extract_event_index(event_dir: Path) -> int | None:
    info_path = event_dir / "event_info.json"
    if not info_path.exists():
        return None
    try:
        payload = json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    index_value = payload.get("event_index")
    if isinstance(index_value, int):
        return index_value
    if isinstance(index_value, str) and index_value.isdigit():
        return int(index_value)
    return None


def event_sort_key(event_dir: Path) -> tuple[int, str]:
    event_index = extract_event_index(event_dir)
    if event_index is None:
        return (10**9, event_dir.name)
    return (event_index, event_dir.name)


def discover_events(events_root: Path, max_events: int) -> tuple[list[EventSelection], dict[str, dict[str, int]]]:
    selections: list[EventSelection] = []
    counts: dict[str, dict[str, int]] = {}

    for category in DEFAULT_CATEGORIES:
        category_dir = events_root / category
        if not category_dir.exists():
            counts[category] = {"discovered": 0, "selected": 0, "missing_category": 1}
            continue

        event_dirs = [
            child
            for child in category_dir.iterdir()
            if child.is_dir() and child.name.startswith("event_")
        ]
        event_dirs.sort(key=event_sort_key)

        selected_dirs = event_dirs[: max_events if max_events >= 0 else len(event_dirs)]
        counts[category] = {
            "discovered": len(event_dirs),
            "selected": len(selected_dirs),
            "missing_category": 0,
        }

        for event_dir in selected_dirs:
            selections.append(
                EventSelection(
                    category=category,
                    source_event_dir=event_dir,
                    event_name=event_dir.name,
                    event_index=extract_event_index(event_dir),
                )
            )

    return selections, counts


def validate_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_top = ["scene", "num_targets", "num_frames", "actor_positions"]
    for key in required_top:
        if key not in payload:
            errors.append(f"missing key: {key}")

    scene = payload.get("scene")
    if not isinstance(scene, dict) or "scale" not in scene:
        errors.append("scene.scale missing")

    num_frames = payload.get("num_frames")
    actor_positions = payload.get("actor_positions")
    num_targets = payload.get("num_targets")

    if not isinstance(actor_positions, list) or len(actor_positions) == 0:
        errors.append("actor_positions must be non-empty list")
    else:
        if isinstance(num_frames, int) and len(actor_positions) != num_frames:
            errors.append("num_frames does not match actor_positions length")

    if isinstance(num_targets, int) and isinstance(actor_positions, list) and actor_positions:
        for frame_idx, frame in enumerate(actor_positions, start=1):
            if not isinstance(frame, list):
                errors.append(f"frame {frame_idx} is not a list")
                continue
            if len(frame) != num_targets:
                errors.append(
                    f"frame {frame_idx} target count {len(frame)} != num_targets {num_targets}"
                )
                break

    return errors


def main() -> int:
    args = parse_args()
    if args.max_events == 0:
        print("--max-events cannot be 0", file=sys.stderr)
        return 2
    if args.sample_stride <= 0:
        print("--sample-stride must be > 0", file=sys.stderr)
        return 2
    if args.num_robots <= 0:
        print("--num-robots must be > 0", file=sys.stderr)
        return 2

    ptz_root = args.ptz_root.resolve()
    mdma_root = args.mdma_root.resolve()

    converter_script = ptz_root / "ros2_ws/src/ptz_videography_sim/datasets/convert_event_folder_to_json.py"
    events_root = ptz_root / "ros2_ws/src/ptz_videography_sim/datasets/soccer/game_one/events"
    dest_root = mdma_root / "experiments/game_one/events"

    if not converter_script.exists():
        print(f"Converter not found: {converter_script}", file=sys.stderr)
        return 2
    if not events_root.exists():
        print(f"Events root not found: {events_root}", file=sys.stderr)
        return 2

    selections, counts = discover_events(events_root, args.max_events)
    dest_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    converted = 0
    skipped = 0
    failed = 0

    for event in selections:
        event_dest_dir = dest_root / event.category / event.event_name
        event_dest_dir.mkdir(parents=True, exist_ok=True)
        output_name = f"{event.event_name}_data.json"
        output_path = event_dest_dir / output_name

        if output_path.exists() and not args.overwrite:
            skipped += 1
            results.append(
                {
                    "category": event.category,
                    "event_name": event.event_name,
                    "event_index": event.event_index,
                    "source_event_dir": str(event.source_event_dir),
                    "output_json": str(output_path),
                    "status": "skipped_exists",
                }
            )
            continue

        cmd = [
            sys.executable,
            str(converter_script),
            "--event-dir",
            str(event.source_event_dir),
            "--output",
            str(output_path),
            "--sample-stride",
            str(args.sample_stride),
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            failed += 1
            results.append(
                {
                    "category": event.category,
                    "event_name": event.event_name,
                    "event_index": event.event_index,
                    "source_event_dir": str(event.source_event_dir),
                    "output_json": str(output_path),
                    "status": "conversion_failed",
                    "stderr": proc.stderr.strip(),
                    "stdout": proc.stdout.strip(),
                }
            )
            continue

        validation_errors: list[str] = []
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            # Ensure planner-compatible camera configuration exists.
            payload["num_robots"] = args.num_robots
            payload["robot_fovs"] = [args.robot_fov] * args.num_robots
            payload["sense_dist"] = args.sense_dist
            output_path.write_text(json.dumps(payload), encoding="utf-8")

            validation_errors = validate_payload(payload)
        except Exception as exc:  # pylint: disable=broad-except
            validation_errors = [f"validation_exception: {exc}"]

        if validation_errors:
            failed += 1
            results.append(
                {
                    "category": event.category,
                    "event_name": event.event_name,
                    "event_index": event.event_index,
                    "source_event_dir": str(event.source_event_dir),
                    "output_json": str(output_path),
                    "status": "validation_failed",
                    "errors": validation_errors,
                    "stdout": proc.stdout.strip(),
                }
            )
            continue

        converted += 1
        results.append(
            {
                "category": event.category,
                "event_name": event.event_name,
                "event_index": event.event_index,
                "source_event_dir": str(event.source_event_dir),
                "output_json": str(output_path),
                "status": "converted",
            }
        )

    manifest = {
        "source_events_root": str(events_root),
        "destination_root": str(dest_root),
        "converter_script": str(converter_script),
        "max_events": args.max_events,
        "sample_stride": args.sample_stride,
        "num_robots": args.num_robots,
        "robot_fov": args.robot_fov,
        "sense_dist": args.sense_dist,
        "counts_by_category": counts,
        "summary": {
            "total_selected": len(selections),
            "converted": converted,
            "skipped": skipped,
            "failed": failed,
        },
        "results": results,
    }

    manifest_path = dest_root / "conversion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote manifest: {manifest_path}")
    print(f"Selected: {len(selections)}")
    print(f"Converted: {converted}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
