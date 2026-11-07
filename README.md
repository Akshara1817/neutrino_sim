# Neutrino Event Visualization Prototype

This project contains Python examples for reading particle-event data and plotting particle kinematics. The `visualize_fermi.py` script fetches one selected event from a registered ROOT dataset, calculates track and missing-momentum values, and displays an interactive 3D plot.

The current sample is a Pythia proton-proton dataset. It is not an inverse beta decay simulation and does not contain reactor neutrino events.

## Setup

Use Python 3.10 or newer. From the project directory, create and activate a virtual environment, then install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the Event Visualization

Select a registered dataset and a zero-based event index:

```bash
python src/visualize_fermi.py \
  --dataset pythia_ppZee_run17emb \
  --event-index 0
```

Event index `0` means the first entry in the ROOT tree. The script checks that the requested event exists and reports the available entry count when the index is out of range. It fetches only the selected event's track momentum branches, rather than loading a fixed group of entries.

The interactive plot uses Plotly's existing `fig.show()` display behavior. The script also prints the selected event's track count, jet count, visible energy, and estimated missing transverse momentum.

## Inputs

Required arguments:

| Argument | Description |
| --- | --- |
| `--dataset` | Dataset name from `datasets.json`. |
| `--event-index` | Zero-based event entry to fetch. |

Optional analysis settings:

| Argument | Default | Description |
| --- | --- | --- |
| `--pt-min` | `0.1` GeV | Minimum transverse momentum for a track to be included. |
| `--mass-hypothesis` | `pi+` | Particle name used to assign a mass to each track. |
| `--jet-radius` | `1.0` | Anti-kT jet radius used by FastJet. |

For example, to select event 125, raise the track threshold to 0.2 GeV, and use a proton mass hypothesis:

```bash
python src/visualize_fermi.py \
  --dataset pythia_ppZee_run17emb \
  --event-index 125 \
  --pt-min 0.2 \
  --mass-hypothesis proton \
  --jet-radius 0.8
```

To see all command-line options:

```bash
python src/visualize_fermi.py --help
```

## Dataset Registry

`datasets.json` maps a dataset name to its remote ROOT file URL and tree name. To add another dataset, add an entry with the same fields:

```json
{
  "example_dataset": {
    "description": "Short description of the sample",
    "url": "https://example.org/sample.root",
    "tree": "Events"
  }
}
```

The selected tree must contain the track momentum branches `Track/Track.mPMomentum[XYZ]`. The current event loader is designed for that schema. Other ROOT schemas require a separate branch mapping.

The remote dataset must be reachable from the machine running the script. The first request may take longer while Uproot reads the ROOT file metadata.

## What the Plot Shows

- Blue vectors show tracks that pass the transverse-momentum threshold.
- The dashed red vector shows estimated missing momentum, calculated as the negative vector sum of the selected tracks.
- Each track is assigned the selected mass hypothesis, which is `pi+` by default.

These are prototype calculations for the registered track dataset. They are not a Fermi-gas model comparison, a nuclear spectral-function calculation, or an IBD event simulation.
