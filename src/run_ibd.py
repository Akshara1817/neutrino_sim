"""Command-line entry point for the standalone IBD example."""

from __future__ import annotations

import argparse
from pathlib import Path

from ibd_simulation import (
    IBD_THRESHOLD,
    generate_events,
    load_spectrum,
    write_events,
    write_plot,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPECTRUM = PROJECT_ROOT / "data" / "u235_huber_spectrum.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a small inverse beta decay (IBD) event sample."
    )
    parser.add_argument(
        "--spectrum",
        type=Path,
        default=DEFAULT_SPECTRUM,
        help=f"Input spectrum CSV (default: {DEFAULT_SPECTRUM.relative_to(PROJECT_ROOT)}).",
    )
    parser.add_argument(
        "--events", type=int, default=10000, help="Number of IBD events to generate."
    )
    parser.add_argument(
        "--seed", type=int, default=12345, help="Random seed for repeatable output."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "ibd_events.csv",
        help="Path for the generated event CSV.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "ibd_spectra.png",
        help="Path for the saved spectrum plot (no web server or GUI is started).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        spectrum = load_spectrum(args.spectrum)
        events = generate_events(spectrum, args.events, args.seed)
        write_events(args.output, events)
        write_plot(args.plot, spectrum, events)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    print(f"Generated {len(events):,} IBD events.")
    print(f"IBD threshold: {IBD_THRESHOLD:.6f} MeV")
    print(f"Input spectrum: {args.spectrum}")
    print(f"Event data: {args.output}")
    print(f"Plot: {args.plot}")
    print("The event sample is relative to the input spectrum and is not an absolute event-rate prediction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
