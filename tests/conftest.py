import sys
from pathlib import Path

# Ensure project root is on sys.path to allow `import agents`, `import utils`, etc.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
