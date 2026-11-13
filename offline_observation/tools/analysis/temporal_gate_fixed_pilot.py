"""194: explicit new task after the preserved 193 entrypoint import failure."""
import sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.analysis.temporal_gate_pilot import main
if __name__=='__main__':main(194)
