# Public-release code audit

## Corrections made

1. **Complete escape tally.** The old stepping action counted a neutron only when it entered the radial scoring shell. Neutrons leaving through either cylinder end cap were omitted. The release records the first boundary crossing from `TargetLogical` to any other volume.
2. **Deterministic execution.** The original default run-manager type could select multithreading, while counters and output names were not merged safely. The release uses `SerialOnly` until accumulables and per-thread file merging are implemented.
3. **Natural-uranium composition.** U-234 was added and the atom fractions now sum to one: U-234 0.0054%, U-235 0.7204%, U-238 99.2742%.
4. **Portable dry-run.** `run_scan.py --dry-run` no longer requires a compiled executable, and generated macros are written explicitly as UTF-8.
5. **Dependency cleanup.** Unused `uproot` and `jsonschema` requirements were removed.
6. **Portable inputs.** Transmutation configs now reference the repository-contained minimal TALYS dataset rather than directories outside the project.

## Archived-result triage

Two result families were found:

- `out_selected_plots`: failed run; U and W values repeat exactly and were excluded;
- `out`: later successful run; material and geometry trends differ and a sanitized subset is retained under `docs/results/reference`.

The retained run still predates correction 1, so its values are useful for interface and trend demonstrations only. A new Geant4 scan is required to establish release results.

## Remaining validation work

- build and run against a documented Geant4 version on Linux;
- regenerate all reference plots with the corrected tally and fixed random seeds;
- compare at least one geometry with an independent tally or published benchmark;
- increase primary counts and report convergence, not only Poisson counting error;
- verify physics-list sensitivity;
- replace or verify scenario half-lives against evaluated nuclear data;
- add a multithread-safe implementation before enabling MT execution.

