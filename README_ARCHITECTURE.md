# IBD MVP Architecture Decisions

## Status

This document records the proposed architecture for a small inverse beta decay (IBD) workflow. It does not describe the physics equations or claim that the workflow has been implemented.

## Decisions

### Keep the IBD workflow separate

The IBD workflow will be implemented as a separate command-line path from the existing Pythia event visualization. The existing event viewer and its data ingestion will remain independent.

### Use a local spectrum file as input

The input spectrum will be a CSV stored with the project. The initial file will contain a tabulation of the published Huber U-235 spectrum model. Its filename and documentation will identify it as a model-derived spectrum, not measured event data.

### Separate input, generation, and presentation

The implementation will keep these responsibilities distinct:

- A spectrum-loading component reads and validates the CSV.
- An IBD generation component consumes the loaded spectrum and produces event records.
- A command-line entry point parses user options and coordinates the workflow.
- A presentation component writes event records to CSV and produces plots.

### Keep generated records portable

Generated event records will be written as CSV so they can be inspected and analyzed with common tools without requiring the plotting interface.

### Keep plots as outputs, not as the data interface

Plots will summarize the input spectrum and generated positron-energy distribution. The generated CSV remains the machine-readable output.

### Record model provenance with the implementation

The code and its user-facing documentation will identify the spectrum source and the IBD model used. Model assumptions and supported input columns will be documented alongside the relevant implementation components.

## Scope boundary

This architecture covers a single-isotope example input and a standalone IBD generation workflow. Reactor fuel composition, oscillations, detector response, and integration with the existing Pythia dataset workflow are outside this initial architecture.
