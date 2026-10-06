# proofnet_kernels

Shared kernel package (ARCHITECTURE §3.4). Implemented in P1.

- `core/` — NumPy + stdlib only; runs on workers (CPython and Pyodide).
- `server/` — backend only (scikit-learn, pandas): finalize, reference, compare.
