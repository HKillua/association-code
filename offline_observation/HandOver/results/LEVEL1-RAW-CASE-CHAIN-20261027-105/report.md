# 105 raw CSV to canonical chain

Fixed postselected scenes; no model, truth-repair, or physical ground-truth claim.

## Scene 3035: target 23, query 56

Mismatch count: 0; reference rows match 103: True

| node | raw rows | duplicate rows | conflicting times | sorted canonical exact | fixed events matched |
|---|---:|---:|---:|---|---|
| 4 | 200 | 0 | 0 | True | True |
| 133 | 308 | 0 | 0 | True | True |

Each selected raw-line mapping and time/position step is retained in the case JSON.

| reference query | side | status | raw rows | matches retained 103 |
|---|---|---|---|---|
| 56 | left | interior | [13785] | True |
| 56 | right | interior | [14272] | True |

## Scene 2819: target 1, query 60

Mismatch count: 0; reference rows match 103: True

| node | raw rows | duplicate rows | conflicting times | sorted canonical exact | fixed events matched |
|---|---:|---:|---:|---|---|
| 13 | 694 | 0 | 0 | True | True |
| 32 | 213 | 0 | 0 | True | True |

Each selected raw-line mapping and time/position step is retained in the case JSON.

| reference query | side | status | raw rows | matches retained 103 |
|---|---|---|---|---|
| 60 | left | interior | [789] | True |
| 60 | right | interior | [800] | True |

## Scene 1043: target 3, query 54

Mismatch count: 0; reference rows match 103: True

| node | raw rows | duplicate rows | conflicting times | sorted canonical exact | fixed events matched |
|---|---:|---:|---:|---|---|
| 0 | 699 | 0 | 0 | True | True |
| 14 | 315 | 0 | 0 | True | True |

Each selected raw-line mapping and time/position step is retained in the case JSON.

| reference query | side | status | raw rows | matches retained 103 |
|---|---|---|---|---|
| 54 | left | interior | [354] | True |
| 54 | right | interior | [367] | True |
| 55 | left | interior | [370] | True |
| 55 | right | interior | [374] | True |

## Limits

- Equality with retained 103 records does not validate physical truth or actual timestamp semantics.
- Whether the reference member is raw AIS or transformed is unknown; source codes cannot be mapped to source1/source2 by assumption.
- Three scenes were selected after observing large errors; no overall anomaly rate or generalization claim.
- Complete original identity, coordinates and raw source rows stay in the server-only private directory.
