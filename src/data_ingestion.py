import skhep_testdata, uproot
import awkward as ak
import particle
from hepunits import GeV
import vector

vector.register_awkward()

picodst = uproot.open(
    "https://zenodo.org/records/21777191/files/pythia_ppZee_run17emb.picoDst.root:PicoDst"
)
px, py, pz = ak.unzip(
    picodst.arrays(filter_name=["Track/Track.mPMomentum[XYZ]"], entry_stop=100)
)

probable_mass = particle.Particle.from_name("pi+").mass / GeV

pseudojets = ak.zip(
    {"px": px, "py": py, "pz": pz, "mass": probable_mass}, with_name="Momentum4D"
)
good_pseudojets = pseudojets[pseudojets.pt > 0.1]

import fastjet

jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 1.0)

clusterseq = fastjet.ClusterSequence(good_pseudojets, jetdef)
clusterseq.inclusive_jets()

x,y=ak.num(good_pseudojets), ak.num(clusterseq.inclusive_jets())
print(x,y)
import numpy as np
sum_px=ak.sum(good_pseudojets.px,axis=1)
sum_py=ak.sum(good_pseudojets.py,axis=1)
sum_pz=ak.sum(good_pseudojets.pz,axis=1)
missing_px=-sum_px
missing_py=-sum_py
missing_pz=-sum_pz

visible_energy = np.sqrt(
    good_pseudojets.px**2 + 
    good_pseudojets.py**2 + 
    good_pseudojets.pz**2 + 
    good_pseudojets.mass**2
)
sum_E = ak.sum(visible_energy, axis=1)

missing_pt = np.sqrt(missing_px**2 + missing_py**2)

missing_p4 = ak.zip(
    {
        "px": missing_px,
        "py": missing_py,
        "pz": missing_pz,
        "mass": ak.zeros_like(missing_px) 
    },
    with_name="Momentum4D"
)

print(f"Missing PT (first 5 events): {missing_pt[:5]}")
print(f"Missing Energy (first 5 events): {missing_p4.energy[:5]}")
