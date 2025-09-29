from importlib import resources as _res

# Ensure packaged lib is on the loader’s search path via rpath $ORIGIN/lib.
# Nothing to do here if you link with -Wl,-rpath,'$ORIGIN/lib' (we will).

# Re-export C++ API from the binary extension:
from .gasalwrap import GasalAligner  # type: ignore
__all__ = ["GasalAligner"]

