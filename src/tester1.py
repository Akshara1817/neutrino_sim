import skhep_testdata
import uproot
import matplotlib.pyplot as plt
import awkward as ak
#previously uproot-Event.root
#import uproot


# If you are on Windows or don't have XRootD installed you can use this url instead
file_url = "https://root.cern/files/rootbench/Run2012BC_DoubleMuParked_Muons.root"
#for mac: file_url = "root://eospublic.cern.ch//eos/opendata/cms/derived-data/AOD2NanoAODOutreachTool/Run2012BC_DoubleMuParked_Muons.root"

file = uproot.open(file_url)
tree = file["Events"]
#tree.show()
# without entry_stop, it will take a long time and it might even fail
#print(tree["nMuon"].array(entry_stop=10_000))
#muons = tree.arrays(["Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "Muon_charge"], entry_stop=1_000)
#print(muons["Muon_pt"])
muons = tree.arrays(entry_stop=1_000)
cut = muons["nMuon"] == 2

pt0 = muons["Muon_pt", cut, 0]
pt1 = muons["Muon_pt", cut, 1]
eta0 = muons["Muon_eta", cut, 0]
eta1 = muons["Muon_eta", cut, 1]
phi0 = muons["Muon_phi", cut, 0]
phi1 = muons["Muon_phi", cut, 1]

import numpy as np

mass = np.sqrt(2 * pt0 * pt1 * (np.cosh(eta0 - eta1) - np.cos(phi0 - phi1)))

import hist

masshist = hist.Hist(hist.axis.Regular(120, 0, 120, label="mass [GeV]"))
masshist.fill(mass)
masshist.plot()
plt.show()

