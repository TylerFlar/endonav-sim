"""Auto-configure SOFA Framework paths for SofaPython3.

Call :func:`init_sofa` before importing any ``Sofa*`` modules. It locates
the SOFA installation relative to this repository (expected at
``../../SOFA/SOFA_v25.12.00_Win64`` or overridden via the ``SOFA_ROOT``
environment variable) and registers the necessary DLL and Python paths.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SOFA_INITIALIZED = False

# Default location relative to this repo: ../SOFA/SOFA_v25.12.00_Win64
_DEFAULT_SOFA_REL = Path(__file__).resolve().parent.parent.parent / "SOFA" / "SOFA_v25.12.00_Win64"


def _find_sofa_root() -> Path:
    """Resolve SOFA_ROOT from environment or the default sibling path."""
    env = os.environ.get("SOFA_ROOT")
    if env:
        p = Path(env)
        if p.is_dir():
            return p
    if _DEFAULT_SOFA_REL.is_dir():
        return _DEFAULT_SOFA_REL
    raise FileNotFoundError(
        "SOFA installation not found. Either set the SOFA_ROOT environment "
        "variable or extract the SOFA binary to "
        f"{_DEFAULT_SOFA_REL}"
    )


def init_sofa() -> Path:
    """Configure Python/DLL paths for SOFA. Returns the SOFA_ROOT path.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _SOFA_INITIALIZED
    if _SOFA_INITIALIZED:
        return _find_sofa_root()

    root = _find_sofa_root()
    site_packages = root / "plugins" / "SofaPython3" / "lib" / "python3" / "site-packages"
    bin_dir = root / "bin"

    if not site_packages.is_dir():
        raise FileNotFoundError(f"SofaPython3 site-packages not found at {site_packages}")

    # Add SofaPython3 to sys.path
    sp = str(site_packages)
    if sp not in sys.path:
        sys.path.insert(0, sp)

    # Register DLL directory (Windows-only, Python 3.8+)
    if hasattr(os, "add_dll_directory") and bin_dir.is_dir():
        os.add_dll_directory(str(bin_dir))
        # Also add plugin bin dirs
        for plugin_bin in root.glob("plugins/*/bin"):
            os.add_dll_directory(str(plugin_bin))

    # Set env vars so SOFA's own init can find itself
    os.environ.setdefault("SOFA_ROOT", str(root))
    os.environ.setdefault(
        "SOFAPYTHON3_ROOT", str(root / "plugins" / "SofaPython3")
    )

    _SOFA_INITIALIZED = True
    return root


__all__ = ["init_sofa"]
