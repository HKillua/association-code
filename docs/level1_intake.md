# Level1 intake and cross-level isolation update — 2026-10-16

The three original Level1 volumes match ScienceDB MD5 values (see `level1_intake_summary.json`). They form one split ZIP; do not extract each volume independently. Local inspection used a seekable, read-only concatenated stream without creating another full-sized archive. No dataset files belong in Git.

Both levels contain 5000 association tables, 5000 measurement CSVs and 5000 reference CSVs. Level1 expands to 8,105,212,552 bytes (7.55 GiB); Level2 to 2,585,704,102 bytes (2.41 GiB). All 10000 association tables were read and CRC checked, but measurement/reference content was not comprehensively audited. There are 26221 shared A identifiers, 5732 exclusive to Level1, and 4584 exclusive to Level2. None of the 5000 equal-numbered scenes has an equal A set. Level names do not establish train/test roles, difficulty, independence, or the level used by TMTAFF.

## Development exposure correction

During initial schema inspection, the first 600 bytes of each Level1 scene 0 CSV (label, measurement, reference) were previewed **before** checking cross-level overlap. This was not a predeclared development scene. Record the exposure and conservatively classify that whole scene as development material.

Its A identifiers occur in these 18 Level2 scenes:

`133, 878, 938, 1174, 1214, 1372, 1443, 1853, 1929, 2052, 2244, 2697, 3048, 3670, 3839, 4576, 4682, 4731`.

All belong to `component-9322e9da0f8c1a13b77c34eb`. The previous seed-20261016 candidate split placed the component in V_select. No actual native-data training/evaluation or production frozen manifest has been reported. The inspection did not read these Level2 scenes' measurements or references.

The revised default development anchors add all 18 scenes to the original 53, yielding 71. Their entire shared-A component is forced to T_core. Manifest/config validation rejects an older manifest missing these anchors. Rebuild preparation into a new output directory; never alter a previously generated manifest or silently reuse pre-correction checkpoints. Removing a component from cyclic assignment may also change other components' assignments.

This correction protects the Level2-only pilot from the known preview. It does **not** authorize combining levels. A combined experiment would need components keyed across `(archive, scene)` and a new protocol. Reference trajectories remain excluded from the association training reader.

## Execution decision

Keep Level2 for the bounded pilot and retain Level1 for later, separately designed experiments. Do not claim Level1 is unnecessary or that using it guarantees a stronger baseline. TMTAFF's actual level remains unknown. The official 4-page competition brief, MD5 `ad67819d5e75b75a16f4eb4bcb003790`, also does not define the levels.

Official source: https://www.scidb.cn/en/detail?dataSetId=c7d8dc56fe854ec2b084d075feb887fd (V2; hash values verified in the preceding website review).
