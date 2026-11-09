from ibd_simulation import load_spectrum, Spectrum, Event, write_events, M_E,M_N, M_P, _sample_cos_theta,positron_energy_mev,differential_cross_section_cm2,total_cross_section_cm2
import numpy as np
import math
from dataclasses import dataclass

@dataclass(frozen=True)
class Event_Fermi(Event):
    fermi_Pep: float
    fermi_Pn: float

def generate_events(
    spectrum: Spectrum, event_count: int, k_Fermi: float, E_binding: float, seed: int |None=None
)-> list[Event]:
    if event_count<1:
        raise ValueError("need event bro")
    rng=np.random.default_rng(seed)
    cross_sections=np.asarray(
        [total_cross_section_cm2(float(e)) for e in spectrum.energy_mev], dtype=float

    )
    weights=spectrum.relative_flux*cross_sections
    if not np.any(weights>0):
        raise ValueError("no energy")
    probs=weights/weights.sum()
    events: list[Event]=[]
    while len(events)<event_count:
        #antineutrino sample
        event_id=len(events)+1
        idx=int(rng.choice(len(spectrum.energy_mev), p=probs))
        enu=float(spectrum.energy_mev[idx])
        #sample positron scatering angle and energy
        cost=_sample_cos_theta(enu,rng)
        ee=positron_energy_mev(enu,cost)
        pe=math.sqrt(max(0.0,ee*ee-M_E*M_E))
        #positron 3 momentum
        phi=float(rng.uniform(0.0,2.0*math.pi))
        transverse=pe*math.sqrt(max(0.0,1.0-cost**2))
        pex=transverse*math.cos(phi)
        pey=transverse*math.sin(phi)
        pez=pe*cost
        #neutron calculations
        up=float(rng.uniform(0.0,1.0))
        pf=k_Fermi*(up**(1.0/3.0))
        cost_p=float(rng.uniform(-1.0,1.0))
        sint_p = math.sqrt(max(0.0, 1.0 - cost_p**2))
        phi_p=float(rng.uniform(0.0,2.0*math.pi))
        ppx=pf*sint_p*math.cos(phi_p)
        ppy=pf*sint_p*math.sin(phi_p)
        ppz=pf*cost_p
        #neutron=neutrino + proton - positron
        pnx=0.0+ppx-pex
        pny=0.0+ppy-pey
        pnz=enu+ppz-pez
        pn_mag=math.sqrt(pnx**2+pny**2+pnz**2)
        neutron_kinetic = math.sqrt(M_N**2 + pnx**2 + pny**2 + pnz**2) - M_N
        if pn_mag<k_Fermi:
            continue
        events.append(
                    Event_Fermi(
                        event_id=event_id,
                        neutrino_energy_mev=enu,
                        cross_section_cm2=float(cross_sections[idx]),
                        positron_total_energy_mev=ee,
                        positron_kinetic_energy_mev=ee - M_E,
                        positron_momentum_mev_c=pe,
                        positron_cos_theta=cost,
                        positron_px_mev_c=pex,
                        positron_py_mev_c=pey,
                        positron_pz_mev_c=pez,
                        neutron_kinetic_energy_mev=neutron_kinetic,
                        neutron_px_mev_c=pnx,
                        neutron_py_mev_c=pny,
                        neutron_pz_mev_c=pnz,
                        fermi_Pep=pf,         # Initial proton momentum magnitude
                        fermi_Pn=pn_mag,        #final calculated neutron momentum magnitude
                    )
                )
    return events



    
    
    
