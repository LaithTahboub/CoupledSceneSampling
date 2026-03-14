import sys
from pathlib import Path

# Add stable-virtual-camera to path so seva imports work
_seva_root = Path(__file__).parent.parent / "stable-virtual-camera"
if str(_seva_root) not in sys.path:
    sys.path.insert(0, str(_seva_root))
