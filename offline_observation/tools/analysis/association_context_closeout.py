"""One read-only output audit followed by complete saved-attribute accounting."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.analysis import audit_association_context_outputs as audit
from tools.analysis.saved_attribute_accounting import account

def closeout(run):
    audit.parent.audit(run,run/'independent-output-audit.json',extra_scene=audit.scene,extra_summary=audit.summary)
    account(run,run/'four-attribute-accounting.json')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();closeout(a.run.resolve())
