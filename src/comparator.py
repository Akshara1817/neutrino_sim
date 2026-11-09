from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import scipy.stats as stats

from fermi_event import Event_Fermi, generate_events as generate_fermi_events
from spectral_event import Event_Spectral, generate_events as generate_spectral_events
from ibd_simulation import load_spectrum, Spectrum, Event


@dataclass(frozen=True)
class KinematicCuts:
    """Phase-space kinematic selection thresholds."""
    min_positron_ke: float = 0.0
    max_positron_ke: float = float("inf")
    min_neutron_ke: float = 0.0
    max_neutron_ke: float = float("inf")
    min_positron_cost: float = -1.0
    max_positron_cost: float = 1.0


def apply_cuts(events: Sequence[Event], cuts: KinematicCuts) -> list[Event]:
    """Filter generated events according to the defined phase-space criteria."""
    filtered: list[Event] = []
    for ev in events:
        if not (cuts.min_positron_ke <= ev.positron_kinetic_energy_mev <= cuts.max_positron_ke):
            continue
        if not (cuts.min_neutron_ke <= ev.neutron_kinetic_energy_mev <= cuts.max_neutron_ke):
            continue
        if not (cuts.min_positron_cost <= ev.positron_cos_theta <= cuts.max_positron_cost):
            continue
        filtered.append(ev)
    return filtered


def compute_statistical_delta(
    fermi_filtered: Sequence[Event_Fermi],
    spectral_filtered: Sequence[Event_Spectral],
    fermi_total_generated: int,
    spectral_total_generated: int,
) -> dict[str, dict[str, float] | float]:
    """
    Computes statistical divergence (KS test and Chi-Square distance)
    and phase-space cut efficiencies for Fermi vs. Spectral Function models.
    """
    # 1. Kinematic arrays
    fermi_e_ke = np.array([e.positron_kinetic_energy_mev for e in fermi_filtered])
    spectral_e_ke = np.array([e.positron_kinetic_energy_mev for e in spectral_filtered])
    
    fermi_cost = np.array([e.positron_cos_theta for e in fermi_filtered])
    spectral_cost = np.array([e.positron_cos_theta for e in spectral_filtered])
    
    fermi_n_ke = np.array([e.neutron_kinetic_energy_mev for e in fermi_filtered])
    spectral_n_ke = np.array([e.neutron_kinetic_energy_mev for e in spectral_filtered])

    # 2. Two-sample Kolmogorov-Smirnov Tests
    ks_results = {
        "positron_ke": stats.ks_2samp(fermi_e_ke, spectral_e_ke),
        "positron_cost": stats.ks_2samp(fermi_cost, spectral_cost),
        "neutron_ke": stats.ks_2samp(fermi_n_ke, spectral_n_ke),
    }

    # 3. Binned Chi-Square Distance for Positron Kinetic Energy
    min_val = min(fermi_e_ke.min(), spectral_e_ke.min()) if len(fermi_e_ke) and len(spectral_e_ke) else 0.0
    max_val = max(fermi_e_ke.max(), spectral_e_ke.max()) if len(fermi_e_ke) and len(spectral_e_ke) else 10.0
    bins = np.linspace(min_val, max_val, 31)

    hist_fermi, _ = np.histogram(fermi_e_ke, bins=bins)
    hist_spectral, _ = np.histogram(spectral_e_ke, bins=bins)

    # Chi-square distance: 0.5 * sum((P - Q)^2 / (P + Q + eps))
    denom = hist_fermi + hist_spectral
    valid_bins = denom > 0
    chi2_dist = 0.5 * np.sum(
        ((hist_fermi[valid_bins] - hist_spectral[valid_bins]) ** 2) / denom[valid_bins]
    )

    # 4. Selection efficiencies
    fermi_eff = len(fermi_filtered) / fermi_total_generated if fermi_total_generated > 0 else 0.0
    spectral_eff = len(spectral_filtered) / spectral_total_generated if spectral_total_generated > 0 else 0.0

    return {
        "cut_efficiency": {
            "fermi": fermi_eff,
            "spectral": spectral_eff,
            "fermi_retained_events": len(fermi_filtered),
            "spectral_retained_events": len(spectral_filtered),
        },
        "ks_tests": {
            "positron_ke_stat": float(ks_results["positron_ke"].statistic),
            "positron_ke_pvalue": float(ks_results["positron_ke"].pvalue),
            "positron_cost_stat": float(ks_results["positron_cost"].statistic),
            "positron_cost_pvalue": float(ks_results["positron_cost"].pvalue),
            "neutron_ke_stat": float(ks_results["neutron_ke"].statistic),
            "neutron_ke_pvalue": float(ks_results["neutron_ke"].pvalue),
        },
        "chi2_distance_e_ke": float(chi2_dist),
    }


def plot_model_comparison(
    fermi_filtered: Sequence[Event_Fermi],
    spectral_filtered: Sequence[Event_Spectral],
    stats_metrics: dict,
) -> go.Figure:
    """Overlays Fermi and Spectral distributions with live statistical annotations."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "Positron Kinetic Energy (MeV)",
            "Positron Scattering Angle cos(θ)",
            "Neutron Kinetic Energy (MeV)",
        )
    )

    # Positron KE
    fig.add_trace(
        go.Histogram(
            x=[e.positron_kinetic_energy_mev for e in fermi_filtered],
            name="Fermi Gas",
            opacity=0.6,
            marker_color="#1f77b4",
            histnorm="probability density",
            nbinsx=40,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=[e.positron_kinetic_energy_mev for e in spectral_filtered],
            name="Spectral Function",
            opacity=0.6,
            marker_color="#ff7f0e",
            histnorm="probability density",
            nbinsx=40,
        ),
        row=1, col=1,
    )

    # Angular distribution cos(theta)
    fig.add_trace(
        go.Histogram(
            x=[e.positron_cos_theta for e in fermi_filtered],
            name="Fermi Gas",
            opacity=0.6,
            marker_color="#1f77b4",
            histnorm="probability density",
            showlegend=False,
            nbinsx=40,
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Histogram(
            x=[e.positron_cos_theta for e in spectral_filtered],
            name="Spectral Function",
            opacity=0.6,
            marker_color="#ff7f0e",
            histnorm="probability density",
            showlegend=False,
            nbinsx=40,
        ),
        row=1, col=2,
    )

    # Neutron KE
    fig.add_trace(
        go.Histogram(
            x=[e.neutron_kinetic_energy_mev for e in fermi_filtered],
            name="Fermi Gas",
            opacity=0.6,
            marker_color="#1f77b4",
            histnorm="probability density",
            showlegend=False,
            nbinsx=40,
        ),
        row=1, col=3,
    )
    fig.add_trace(
        go.Histogram(
            x=[e.neutron_kinetic_energy_mev for e in spectral_filtered],
            name="Spectral Function",
            opacity=0.6,
            marker_color="#ff7f0e",
            histnorm="probability density",
            showlegend=False,
            nbinsx=40,
        ),
        row=1, col=3,
    )

    # Annotation badge for statistical metrics
    ks = stats_metrics["ks_tests"]
    eff = stats_metrics["cut_efficiency"]
    annotation_text = (
        f"<b>Statistical Delta (Post-Cuts)</b><br>"
        f"Retained Events (Fermi/SF): {eff['fermi_retained_events']} / {eff['spectral_retained_events']}<br>"
        f"Efficiency (Fermi/SF): {eff['fermi']*100:.1f}% / {eff['spectral']*100:.1f}%<br>"
        f"Positron KE KS Stat: {ks['positron_ke_stat']:.4f} (p={ks['positron_ke_pvalue']:.2e})<br>"
        f"Neutron KE KS Stat: {ks['neutron_ke_stat']:.4f} (p={ks['neutron_ke_pvalue']:.2e})<br>"
        f"Positron KE χ² Distance: {stats_metrics['chi2_distance_e_ke']:.2f}"
    )

    fig.update_layout(
        barmode="overlay",
        title_text="Nuclear Model Comparative Diagnostic: Fermi Gas vs. Spectral Function",
        annotations=[
            dict(
                x=0.01,
                y=1.18,
                xref="paper",
                yref="paper",
                text=annotation_text,
                showarrow=False,
                align="left",
                bgcolor="rgba(240, 240, 240, 0.8)",
                bordercolor="gray",
                borderwidth=1,
            )
        ],
        margin=dict(t=100, l=40, r=40, b=40),
    )
    return fig


if __name__ == "__main__":
    from pathlib import Path

    spectrum_path = Path("tests/data/u235_spectrum.csv")
    if not spectrum_path.exists():
        # Fallback dummy spectrum for standalone testing
        energies = np.linspace(1.9, 9.0, 50)
        fluxes = np.exp(-energies / 2.0)
        spec = Spectrum(energy_mev=energies, relative_flux=fluxes)
    else:
        spec = load_spectrum(spectrum_path)

    n_events = 2000
    k_f = 225.0  # Fermi momentum in MeV/c (e.g., Carbon-12)
    e_b = 25.0   # Binding energy in MeV

    # Generate from both models
    print(f"Generating {n_events} Fermi gas events...")
    fermi_evs = generate_fermi_events(spec, n_events, k_Fermi=k_f, E_binding=e_b, seed=42)

    print(f"Generating {n_events} Spectral Function events...")
    spectral_evs = generate_spectral_events(spec, n_events, k_Fermi=k_f, seed=42)

    # Define user phase-space cuts (e.g. sensor detection thresholds)
    user_cuts = KinematicCuts(
        min_positron_ke=1.5,
        max_positron_ke=6.0,
        min_positron_cost=-0.8,
        max_positron_cost=0.8,
    )

    fermi_sliced = apply_cuts(fermi_evs, user_cuts)
    spectral_sliced = apply_cuts(spectral_evs, user_cuts)

    # Compute Kolmogorov-Smirnov and Chi-Square metrics
    metrics = compute_statistical_delta(fermi_sliced, spectral_sliced, n_events, n_events)

    print("\n--- Quantitative Model Metrics ---")
    print(f"Fermi Cut Efficiency: {metrics['cut_efficiency']['fermi']*100:.2f}%")
    print(f"Spectral Cut Efficiency: {metrics['cut_efficiency']['spectral']*100:.2f}%")
    print(f"Positron KE KS-statistic: {metrics['ks_tests']['positron_ke_stat']:.4f} (p-val: {metrics['ks_tests']['positron_ke_pvalue']:.3e})")
    print(f"Positron KE Chi-Square Distance: {metrics['chi2_distance_e_ke']:.3f}")

    # Render multi-panel diagnostic
    fig = plot_model_comparison(fermi_sliced, spectral_sliced, metrics)
    fig.show()