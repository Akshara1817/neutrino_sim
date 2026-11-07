# Standalone IBD Example

This adds a small, separate inverse beta decay workflow. It does not alter the existing Pythia event viewer.

## Model inputs

The default spectrum file, `data/u235_huber_spectrum.csv`, contains a tabulation of the U-235 antineutrino flux model of Huber and Schwetz. It is generated from their published six-coefficient parameterization and is model-derived input, not measured event data. The cited source fits spectra inferred from measured beta spectra. The tabulation covers 2 to 8 MeV at 0.1 MeV intervals.

The IBD calculation uses the first-order-in-1/M differential cross section and positron-energy relation from Vogel and Beacom. It assumes a stationary free proton, includes their energy-independent inner radiative normalization, and omits energy-dependent outer radiative corrections. It does not model reactor fuel composition, neutrino oscillations, detector response, or backgrounds.

References:

- P. Huber and T. Schwetz, “Precision spectroscopy with reactor anti-neutrinos,” *Physical Review D* 70, 053011 (2004), [arXiv:hep-ph/0407026](https://arxiv.org/abs/hep-ph/0407026).
- P. Vogel and J. F. Beacom, “The angular distribution of the reaction anti-nu_e + p -> e+ + n,” *Physical Review D* 60, 053003 (1999), [arXiv:hep-ph/9903554](https://arxiv.org/abs/hep-ph/9903554).

## Run

From the project root, using the project virtual environment:

```bash
.venv/bin/python src/run_ibd.py
```

The command writes `outputs/ibd_events.csv` and `outputs/ibd_spectra.png`. It writes the plot to a file and does not launch a browser, GUI, or local server.

Options:

```text
--spectrum PATH   Input CSV with energy_mev,relative_flux columns
--events N        Number of events to generate (default: 10000)
--seed N          Random seed (default: 12345)
--output PATH     Generated event CSV path
--plot PATH       Plot image path
```

For example:

```bash
.venv/bin/python src/run_ibd.py --events 5000 --seed 7 \
  --output outputs/sample.csv --plot outputs/sample.png
```

The generated events are sampled from the supplied relative spectrum, weighted by the IBD cross section. The event count is user-selected, so the output is a shape-level Monte Carlo sample, not a predicted physical rate. Event energies are sampled at the energy points in the spectrum CSV. The CSV stores antineutrino energy, positron and neutron kinematics, and the total cross section evaluated at the selected antineutrino energy.

The CSV reader requires finite, strictly increasing positive energies, nonnegative relative flux values, and at least two rows. Energies below the IBD threshold contribute zero event weight.

## Dependencies

The command uses NumPy for sampling and Matplotlib for the saved plot. Both are already listed in the project `requirements.txt`.
