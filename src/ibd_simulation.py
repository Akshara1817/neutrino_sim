"""Small IBD event generator using a tabulated antineutrino spectrum.

The default CSV is tabulated from the U-235 parameterization in Huber and
Schwetz, Phys. Rev. D 70, 053011 (2004), arXiv:hep-ph/0407026. The IBD
kinematics and differential cross section use the O(1/M) treatment in Vogel
and Beacom, Phys. Rev. D 60, 053003 (1999), arXiv:hep-ph/9903554.
"""

from __future__ import annotations

import csv
from functools import lru_cache
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


# Masses are in MeV/c^2. Energies and momenta are represented in MeV and MeV/c.
M_E = 0.51099895000
M_P = 938.27208816
M_N = 939.56542052
M_NUCLEON = (M_P + M_N) / 2.0
DELTA = M_N - M_P
IBD_THRESHOLD = ((M_N + M_E) ** 2 - M_P**2) / (2.0 * M_P)

F = 1.0
G = 1.26
F2 = 3.706
# The coefficient gives the O(1) total cross section in cm^2 when energies
# are in MeV: sigma = 0.0952e-42 * E_e * p_e.
SIGMA0_CM2_PER_MEV2 = 0.0952e-42 / (F**2 + 3.0 * G**2)


@dataclass(frozen=True)
class Spectrum:
    energy_mev: np.ndarray
    relative_flux: np.ndarray


@dataclass(frozen=True)
class Event:
    event_id: int
    neutrino_energy_mev: float
    cross_section_cm2: float
    positron_total_energy_mev: float
    positron_kinetic_energy_mev: float
    positron_momentum_mev_c: float
    positron_cos_theta: float
    positron_px_mev_c: float
    positron_py_mev_c: float
    positron_pz_mev_c: float
    neutron_kinetic_energy_mev: float
    neutron_px_mev_c: float
    neutron_py_mev_c: float
    neutron_pz_mev_c: float


def load_spectrum(path: Path) -> Spectrum:
    """Load a two-column CSV with energy_mev and relative_flux columns."""
    energies: list[float] = []
    fluxes: list[float] = []
    try:
        with path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            required = {"energy_mev", "relative_flux"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(
                    "Spectrum CSV must include energy_mev and relative_flux columns."
                )
            for row_number, row in enumerate(reader, start=2):
                try:
                    energy = float(row["energy_mev"])
                    flux = float(row["relative_flux"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid numeric value on CSV row {row_number}.") from exc
                if not math.isfinite(energy) or not math.isfinite(flux):
                    raise ValueError(f"Non-finite value on CSV row {row_number}.")
                if energy <= 0 or flux < 0:
                    raise ValueError(
                        f"Energy must be positive and flux nonnegative on row {row_number}."
                    )
                energies.append(energy)
                fluxes.append(flux)
    except OSError as exc:
        raise ValueError(f"Could not read spectrum file: {path}") from exc

    energy_array = np.asarray(energies, dtype=float)
    flux_array = np.asarray(fluxes, dtype=float)
    if energy_array.size < 2:
        raise ValueError("Spectrum CSV must contain at least two data rows.")
    if np.any(np.diff(energy_array) <= 0):
        raise ValueError("Spectrum energies must be strictly increasing.")
    if not np.any(flux_array > 0):
        raise ValueError("Spectrum must contain at least one positive flux value.")
    return Spectrum(energy_array, flux_array)


def positron_energy_mev(neutrino_energy_mev: float, cos_theta: float) -> float:
    """Return the Vogel-Beacom first-order positron total energy."""
    e0 = neutrino_energy_mev - DELTA
    if e0 <= M_E:
        return float("nan")
    p0 = math.sqrt(max(0.0, e0 * e0 - M_E * M_E))
    v0 = p0 / e0
    y2 = (DELTA * DELTA - M_E * M_E) / 2.0
    return e0 * (
        1.0 - neutrino_energy_mev / M_NUCLEON * (1.0 - v0 * cos_theta)
    ) - y2 / M_NUCLEON


def differential_cross_section_cm2(
    neutrino_energy_mev: float, cos_theta: float
) -> float:
    """Vogel-Beacom dσ/dcos(theta), through first order in 1/M.

    The energy-independent inner radiative correction is included in the
    normalization (0.0952e-42 cm^2/MeV^2). Energy-dependent outer radiative
    corrections are not included.
    """
    if not IBD_THRESHOLD < neutrino_energy_mev or not -1.0 <= cos_theta <= 1.0:
        return 0.0
    e0 = neutrino_energy_mev - DELTA
    if e0 <= M_E:
        return 0.0
    p0 = math.sqrt(max(0.0, e0 * e0 - M_E * M_E))
    v0 = p0 / e0
    e1 = positron_energy_mev(neutrino_energy_mev, cos_theta)
    if e1 <= M_E:
        return 0.0
    p1 = math.sqrt(max(0.0, e1 * e1 - M_E * M_E))
    v1 = p1 / e1
    c = cos_theta

    gamma = (
        2.0
        * (F + F2)
        * G
        * ((2.0 * e0 + DELTA) * (1.0 - v0 * c) - M_E**2 / e0)
        + (F**2 + G**2) * (DELTA * (1.0 + v0 * c) + M_E**2 / e0)
        + (F**2 + 3.0 * G**2)
        * ((e0 + DELTA) * (1.0 - c / v0) - DELTA)
        + (F**2 - G**2)
        * ((e0 + DELTA) * (1.0 - c / v0) - DELTA)
        * v0
        * c
    )
    leading = ((F**2 + 3.0 * G**2) + (F**2 - G**2) * v1 * c) * e1 * p1
    first_order_term = gamma / M_NUCLEON * e0 * p0
    result = SIGMA0_CM2_PER_MEV2 * 0.5 * (leading - first_order_term)
    return max(0.0, result)


def total_cross_section_cm2(neutrino_energy_mev: float) -> float:
    """Numerically integrate the differential cross section over cos(theta)."""
    if neutrino_energy_mev <= IBD_THRESHOLD:
        return 0.0
    cosines = np.linspace(-1.0, 1.0, 401)
    values = np.asarray(
        [differential_cross_section_cm2(neutrino_energy_mev, float(c)) for c in cosines]
    )
    return float(np.sum((values[:-1] + values[1:]) * np.diff(cosines) / 2.0))


@lru_cache(maxsize=512)
def _angular_grid(neutrino_energy_mev: float) -> tuple[np.ndarray, np.ndarray, float]:
    grid = np.linspace(-1.0, 1.0, 1001)
    density = np.asarray(
        [differential_cross_section_cm2(neutrino_energy_mev, float(c)) for c in grid]
    )
    return grid, density, float(density.max())


def _sample_cos_theta(neutrino_energy_mev: float, rng: np.random.Generator) -> float:
    """Rejection-sample the positron angle from dσ/dcos(theta)."""
    _, _, maximum = _angular_grid(neutrino_energy_mev)
    if maximum <= 0:
        raise ValueError("The selected antineutrino energy is below IBD threshold.")
    ceiling = maximum * 1.000001
    while True:
        cosine = float(rng.uniform(-1.0, 1.0))
        value = differential_cross_section_cm2(neutrino_energy_mev, cosine)
        if rng.uniform(0.0, ceiling) <= value:
            return cosine


def generate_events(
    spectrum: Spectrum, event_count: int, seed: int | None = None
) -> list[Event]:
    """Sample spectrum energies weighted by IBD cross section, then angles."""
    if event_count < 1:
        raise ValueError("Event count must be at least 1.")
    rng = np.random.default_rng(seed)
    cross_sections = np.asarray(
        [total_cross_section_cm2(float(e)) for e in spectrum.energy_mev], dtype=float
    )
    weights = spectrum.relative_flux * cross_sections
    if not np.any(weights > 0):
        raise ValueError("The input spectrum contains no energies above IBD threshold.")
    probabilities = weights / weights.sum()

    events: list[Event] = []
    for event_id in range(1, event_count + 1):
        index = int(rng.choice(len(spectrum.energy_mev), p=probabilities))
        enu = float(spectrum.energy_mev[index])
        cosine = _sample_cos_theta(enu, rng)
        ee = positron_energy_mev(enu, cosine)
        pe = math.sqrt(max(0.0, ee * ee - M_E * M_E))
        phi = float(rng.uniform(0.0, 2.0 * math.pi))
        transverse = pe * math.sqrt(max(0.0, 1.0 - cosine * cosine))
        pex = transverse * math.cos(phi)
        pey = transverse * math.sin(phi)
        pez = pe * cosine
        # The incident antineutrino travels along +z; target proton is at rest.
        pnx, pny, pnz = -pex, -pey, enu - pez
        neutron_kinetic = math.sqrt(M_N**2 + pnx**2 + pny**2 + pnz**2) - M_N
        events.append(
            Event(
                event_id=event_id,
                neutrino_energy_mev=enu,
                cross_section_cm2=float(cross_sections[index]),
                positron_total_energy_mev=ee,
                positron_kinetic_energy_mev=ee - M_E,
                positron_momentum_mev_c=pe,
                positron_cos_theta=cosine,
                positron_px_mev_c=pex,
                positron_py_mev_c=pey,
                positron_pz_mev_c=pez,
                neutron_kinetic_energy_mev=neutron_kinetic,
                neutron_px_mev_c=pnx,
                neutron_py_mev_c=pny,
                neutron_pz_mev_c=pnz,
            )
        )
    return events


def write_events(path: Path, events: Sequence[Event]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(Event.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(event.__dict__ for event in events)


def write_plot(path: Path, spectrum: Spectrum, events: Sequence[Event]) -> None:
    """Save normalized input-flux and generated positron-energy distributions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].plot(spectrum.energy_mev, spectrum.relative_flux, color="#245b8f")
    axes[0].set(title="Input U-235 antineutrino spectrum", xlabel="Antineutrino energy (MeV)", ylabel="Relative flux")
    axes[0].grid(alpha=0.25)

    positron_kinetic = [event.positron_kinetic_energy_mev for event in events]
    axes[1].hist(positron_kinetic, bins=50, density=True, color="#d07836", alpha=0.82)
    axes[1].set(title="Generated positron spectrum", xlabel="Positron kinetic energy (MeV)", ylabel="Normalized events / MeV")
    axes[1].grid(alpha=0.25)

    fig.suptitle("IBD model prediction, free proton target")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
