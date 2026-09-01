# TALYS helper workflow

The scripts in this directory were recovered from the related `npc` project folder and made portable for this repository.

1. Install TALYS 2.0 and make the `talys` executable available on `PATH`.
2. Run `bash tools/talys/run_talys_batch.sh` from any working directory.
3. New runs are written below `tools/talys/runs`; existing run directories are never deleted automatically.
4. Use `python tools/talys/extract_talys_outputs.py --help` to aggregate full outputs.

The smaller tables under `analysis/data/talys` are the curated inputs used by the example transmutation configs.

