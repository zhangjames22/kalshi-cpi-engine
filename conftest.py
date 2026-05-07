"""Make `src/` importable in tests without a full package install."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
