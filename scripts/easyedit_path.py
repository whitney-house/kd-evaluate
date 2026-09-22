"""
Shared helper: makes `import easyeditor` work from scripts that live in
this repo's own scripts/ folder, without copying them into EasyEdit/.

Import this before importing anything from easyeditor:
    import easyedit_path  # noqa: F401  (side effect: adds EasyEdit/ to sys.path)
    from easyeditor import BaseEditor, ROMEHyperParams
"""

import os
import sys

# EasyEdit/ is expected to sit next to this repo's own scripts/ and data/
# folders (i.e. as a sibling directory under the project root), matching
# what setup.sh clones. Override with the EASYEDIT_DIR env var if you keep
# it somewhere else.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
EASYEDIT_DIR = os.environ.get("EASYEDIT_DIR", os.path.join(_PROJECT_ROOT, "EasyEdit"))

if EASYEDIT_DIR not in sys.path:
    sys.path.insert(0, EASYEDIT_DIR)