"""191: exactly the189 mean-velocity expression, now on frozen V32."""
import argparse,sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.analysis.drift_memory_pilot import run
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('parent-run','persistence-run','qualification-run','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--time-policy',choices=('advisory',),required=True);run(parser.parse_args(),split='V_select',number=191)
