"""Load one selected event and calculate its tracks and missing momentum."""

import json
from pathlib import Path

import awkward as ak
import numpy as np
import particle
import uproot
import vector
from hepunits import GeV

import fastjet

vector.register_awkward()

DATASETS_FILE = Path(__file__).resolve().parent.parent / "datasets.json"
TRACK_BRANCHES = ["Track/Track.mPMomentum[XYZ]"]


def read_dataset_registry(path=DATASETS_FILE):
    """Read named ROOT datasets from the project registry."""
    with Path(path).open(encoding="utf-8") as registry_file:
        datasets = json.load(registry_file)

    if not isinstance(datasets, dict) or not datasets:
        raise ValueError(f"Dataset registry is empty or invalid: {path}")
    return datasets


def load_event(
    dataset_name,
    event_index,
    *,
    pt_min=0.1,
    mass_hypothesis="pi+",
    jet_radius=1.0,
    datasets_file=DATASETS_FILE,
):
    """Fetch and prepare one event from the selected registered ROOT dataset.

    The returned track and missing-momentum arrays retain a one-event outer
    dimension, so callers can index them with zero for plotting.
    """
    if event_index < 0:
        raise ValueError("Event index must be zero or greater.")
    if pt_min < 0:
        raise ValueError("The transverse-momentum threshold cannot be negative.")
    if jet_radius <= 0:
        raise ValueError("The jet radius must be greater than zero.")

    datasets = read_dataset_registry(datasets_file)
    if dataset_name not in datasets:
        available = ", ".join(sorted(datasets))
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available datasets: {available}"
        )

    dataset = datasets[dataset_name]
    url = dataset["url"]
    tree_name = dataset["tree"]

    root_file = uproot.open(url)
    try:
        try:
            tree = root_file[tree_name]
        except KeyError as exc:
            raise ValueError(
                f"Tree '{tree_name}' was not found in dataset '{dataset_name}'."
            ) from exc

        entry_count = tree.num_entries
        if event_index >= entry_count:
            raise ValueError(
                f"Event index {event_index} is out of range. Dataset "
                f"'{dataset_name}' contains {entry_count} entries, indexed "
                f"from 0 to {entry_count - 1}."
            )

        try:
            arrays = tree.arrays(
                filter_name=TRACK_BRANCHES,
                entry_start=event_index,
                entry_stop=event_index + 1,
                library="ak",
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Dataset '{dataset_name}' does not contain the expected track "
                "momentum branches."
            ) from exc
    finally:
        root_file.close()

    if len(arrays) != 1:
        raise ValueError(
            f"Could not read event {event_index} from dataset '{dataset_name}'."
        )

    try:
        px, py, pz = ak.unzip(arrays)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"The track momentum branches in '{dataset_name}' have an "
            "unexpected structure."
        ) from exc

    probable_mass = particle.Particle.from_name(mass_hypothesis).mass / GeV
    pseudojets = ak.zip(
        {"px": px, "py": py, "pz": pz, "mass": probable_mass},
        with_name="Momentum4D",
    )
    good_pseudojets = pseudojets[pseudojets.pt > pt_min]

    track_count = int(ak.num(good_pseudojets, axis=1)[0])
    if track_count:
        jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, jet_radius)
        clusterseq = fastjet.ClusterSequence(good_pseudojets, jetdef)
        jets = clusterseq.inclusive_jets()
        jet_count = int(ak.num(jets, axis=1)[0])
    else:
        jet_count = 0

    sum_px = ak.sum(good_pseudojets.px, axis=1)
    sum_py = ak.sum(good_pseudojets.py, axis=1)
    sum_pz = ak.sum(good_pseudojets.pz, axis=1)
    missing_px = -sum_px
    missing_py = -sum_py
    missing_pz = -sum_pz

    visible_energy = np.sqrt(
        good_pseudojets.px**2
        + good_pseudojets.py**2
        + good_pseudojets.pz**2
        + good_pseudojets.mass**2
    )
    sum_energy = ak.sum(visible_energy, axis=1)
    missing_pt = np.sqrt(missing_px**2 + missing_py**2)

    missing_p4 = ak.zip(
        {
            "px": missing_px,
            "py": missing_py,
            "pz": missing_pz,
            "mass": ak.zeros_like(missing_px),
        },
        with_name="Momentum4D",
    )

    return {
        "tracks": good_pseudojets,
        "missing_p4": missing_p4,
        "track_count": track_count,
        "jet_count": jet_count,
        "entry_count": entry_count,
        "sum_energy": float(sum_energy[0]),
        "missing_pt": float(missing_pt[0]),
    }
