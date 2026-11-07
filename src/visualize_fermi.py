"""Visualize tracks and estimated missing momentum for one ROOT event."""

import argparse

import plotly.graph_objects as go
from particle import InvalidParticle, ParticleNotFound

from data_ingestion import load_event, read_dataset_registry


def nonnegative_float(value):
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Value must be zero or greater.")
    return number


def positive_float(value):
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero.")
    return number


def build_parser():
    datasets = read_dataset_registry()
    parser = argparse.ArgumentParser(
        description="Fetch and visualize one event from a registered ROOT dataset."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(datasets),
        help="dataset name from datasets.json",
    )
    parser.add_argument(
        "--event-index",
        required=True,
        type=int,
        help="zero-based event index to fetch from the dataset",
    )
    parser.add_argument(
        "--pt-min",
        type=nonnegative_float,
        default=0.1,
        metavar="GEV",
        help="minimum track transverse momentum (default: 0.1 GeV)",
    )
    parser.add_argument(
        "--mass-hypothesis",
        default="pi+",
        help="particle name used as the track mass hypothesis (default: pi+)",
    )
    parser.add_argument(
        "--jet-radius",
        type=positive_float,
        default=1.0,
        help="anti-kT jet radius (default: 1.0)",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.event_index < 0:
        parser.error("--event-index must be zero or greater")

    try:
        event = load_event(
            args.dataset,
            args.event_index,
            pt_min=args.pt_min,
            mass_hypothesis=args.mass_hypothesis,
            jet_radius=args.jet_radius,
        )
    except (OSError, ValueError, KeyError, InvalidParticle, ParticleNotFound) as exc:
        parser.error(str(exc))

    print(
        f"Dataset: {args.dataset} | Event: {args.event_index} | "
        f"Entries: {event['entry_count']} | "
        f"Tracks: {event['track_count']} | Jets: {event['jet_count']}"
    )
    print(f"Visible energy: {event['sum_energy']:.3g} GeV")
    print(f"Estimated missing PT: {event['missing_pt']:.3g} GeV")

    tracks = event["tracks"][0]
    missing_p4 = event["missing_p4"][0]

    vis_px = tracks.px
    vis_py = tracks.py
    vis_pz = tracks.pz
    miss_px = missing_p4.px
    miss_py = missing_p4.py
    miss_pz = missing_p4.pz

    fig = go.Figure()

    for i in range(len(vis_px)):
        fig.add_trace(
            go.Scatter3d(
                x=[0, vis_px[i]],
                y=[0, vis_py[i]],
                z=[0, vis_pz[i]],
                mode="lines+markers",
                line=dict(color="blue", width=4),
                marker=dict(size=4, symbol="diamond"),
                name=f"Visible Track {i}",
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=[0, miss_px],
            y=[0, miss_py],
            z=[0, miss_pz],
            mode="lines+markers",
            line=dict(color="red", width=6, dash="dash"),
            marker=dict(size=6, symbol="cross"),
            name="Missing Momentum (MET)",
        )
    )

    fig.update_layout(
        title=f"Track and missing momentum (Event {args.event_index})",
        scene=dict(
            xaxis_title="Px (GeV)",
            yaxis_title="Py (GeV)",
            zaxis_title="Pz (GeV)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    fig.show()


if __name__ == "__main__":
    main()
