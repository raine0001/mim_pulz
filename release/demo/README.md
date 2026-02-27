# Demo/UI Release Package

This package runs the local demonstration UI and exposes the full method trace:

`artifact -> structure profile -> routing policy -> candidate evidence -> source trace`

## Install

From project root:

```bash
python -m pip install -r release/demo/requirements.txt
```

## Run

```bash
python app/corpus/query_api.py
```

Open in browser:

- `http://127.0.0.1:50001/corpus/mim`
- `http://127.0.0.1:50001/corpus/help`

## One-Command Wrappers

- POSIX shell: `release/demo/run_demo.sh`
- PowerShell: `release/demo/run_demo.ps1`

## Notes

- Demo uses the same routing and reranker artifacts pinned in `release/manifest.json`.
- If no local image scan exists for a source, use:
  - `Open visual candidates`
  - `Compare source vs visual`
  - `Open source record`

