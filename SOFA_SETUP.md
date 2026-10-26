# SOFA Framework Setup

endonav-sim uses [SOFA Framework](https://www.sofa-framework.org/) for physics simulation (deformable kidney tissue, flexible beam scope, collision/friction contact). SOFA is **not** a pip package — it must be installed separately.

## Quick Setup (Windows)

1. **Download** the SOFA v25.12 pre-built binary:
   ```
   https://github.com/sofa-framework/sofa/releases/download/v25.12.00/SOFA_v25.12.00_Win64.zip
   ```

2. **Extract** to the sibling directory (next to this repo):
   ```
   arclab/
     SOFA/
       SOFA_v25.12.00_Win64/    <-- extract here
         bin/
         plugins/
         ...
     endonav-sim/               <-- this repo
   ```

3. **Verify** Python 3.12 is your active interpreter (SOFA v25.12 bundles SofaPython3 for Python 3.12):
   ```bash
   uv run python --version
   # Should show Python 3.12.x
   ```

4. **Test** the import:
   ```bash
   uv run python -c "from endonav_sim.sofa_setup import init_sofa; init_sofa(); import Sofa; print('SOFA OK')"
   ```

The `endonav_sim.sofa_setup.init_sofa()` function automatically locates SOFA at `../../SOFA/SOFA_v25.12.00_Win64` relative to the package, registers DLL paths, and adds SofaPython3 to `sys.path`. No environment variables needed.

## Custom SOFA Location

If SOFA is installed elsewhere, set the `SOFA_ROOT` environment variable:
```bash
export SOFA_ROOT=/path/to/SOFA_v25.12.00_Win64
```

## Included Plugins

The SOFA v25.12 binary includes all required plugins:
- **BeamAdapter** — flexible beam mechanics for the ureteroscope shaft
- **SoftRobots** — cable/tendon actuator constraints
- **SofaPython3** — Python bindings

## Linux / macOS

Download the appropriate binary from [SOFA releases](https://github.com/sofa-framework/sofa/releases) and extract to the same relative path. The `init_sofa()` function works cross-platform.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'SofaRuntime'`**: `init_sofa()` was not called before importing Sofa modules. The simulator handles this automatically.
- **DLL load failures on Windows**: Ensure you're using Python 3.12 (matching the SOFA binary). Check with `uv run python --version`.
- **SOFA not found**: Verify the directory structure matches the expected path, or set `SOFA_ROOT`.
