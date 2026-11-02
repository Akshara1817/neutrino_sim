import uproot
import awkward as ak
import matplotlib.pyplot as plt



#file_url = "root://eospublic.cern.ch//eos/opendata/cms/derived-data/AOD2NanoAODOutreachTool/Run2012BC_DoubleMuParked_Muons.root"

# If you are on Windows or don't have XRootD installed you can use this url instead
file_url = "https://root.cern/files/rootbench/Run2012BC_DoubleMuParked_Muons.root"

file = uproot.open(file_url)
tree = file["Events"]

arrays = tree.arrays(filter_name="/Muon_(pt|eta|phi|charge)/", entry_stop=10000)

muons = ak.zip(
    {
        "pt": arrays["Muon_pt"],
        "eta": arrays["Muon_eta"],
        "phi": arrays["Muon_phi"],
        "charge": arrays["Muon_charge"],
    }
)
pairs = ak.combinations(muons, 2)
mu1, mu2 = ak.unzip(pairs)
import numpy as np

mass = np.sqrt(
    2 * mu1.pt * mu2.pt * (np.cosh(mu1.eta - mu2.eta) - np.cos(mu1.phi - mu2.phi))
)
import hist
"""
cut=(mu1.charge!=mu2.charge)
hist.Hist(hist.axis.Regular(120, 0, 120, label="mass [GeV]")).fill(
    ak.ravel(mass[cut])
).plot()
plt.show()
"""
import particle, hepunits

zmass = particle.Particle.findall("Z0")[0].mass / hepunits.GeV
which = ak.argmin(abs(mass - zmass), axis=1, keepdims=True)

hist.Hist(hist.axis.Regular(120, 0, 120, label="mass [GeV]")).fill(
    ak.drop_none(ak.ravel(mass[which]))
).plot()
plt.show()