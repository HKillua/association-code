# CDF1 data and registration contract

This is an independent **MTAD-Level2 A-B identity-pair pilot**. It is not the official MTAD V4 output protocol or a claim to reproduce every undocumented TMTAFF preprocessing choice.

## Native data and identities

`association/data.py` opens the native ZIP directly. It recognizes UTF-8 and legacy GBK member names and UTF-8-BOM/GB18030 CSV content. It rejects unsafe names, repeated ZIP members, ambiguous scene aliases, incompatible headers, and unequal label/measurement scene sets. Nothing is extracted. **Reference CSV members are never opened.** The archive's binary SHA-256 is computed for input provenance without parsing reference content.

A node is `(scene, source, batch)`. Exactly one full `A-B-C` label identity must describe the node; conflicting full labels quarantine it even if they share A-B. The supervised identity is **A-B**, and **A** is used for cross-scene leakage grouping. Full labels remain in audit records. Same-source fragments may be positive pairs. Source, batch, identity, label interval, and reference state are not motion features. Labels do not crop measurements to their interval: out-of-label-interval observations are a diagnostic count only.

| Field | Declared representation | Validation / conversion |
| --- | --- | --- |
| time | Native elapsed seconds | Finite, at least two distinct timestamps per node; sorted |
| lat / lon | Degrees | Finite, latitude within ±90 and longitude within ±180; no geodesic conversion |
| vel | Native MTAD velocity | Finite and nonnegative; **no assertion or conversion to m/s or knots** |
| cou | Degrees | -1 is missing; accepted range 0–360, 360 becomes 0; circular interpolation |

Velocity's physical scale is not used to extrapolate positions or form a physics gate. All branches receive the same native value followed by their training-side normalizer. Physical fusion needs separately verified velocity units before it can be added.

Any nonfinite/nonnumeric measurement, negative velocity, missing/out-of-range course, invalid coordinates, missing/ambiguous label, or conflicting values at the same timestamp quarantines the **whole node**. Exactly repeated numeric observations are removed. A single remaining timestamp is insufficient. Same-time 0° and 360° observations are not silently merged: conflict checking precedes angular canonicalization. Quarantine records retain all applicable reasons, affected rows/identities, and positive identity-edge coverage before/after. Unknown measurements or missing measurement nodes are reported rather than silently omitted.

## Split isolation and provenance

`build_manifest(archive, seed, caps, development_scenes)` reads every association table and connects any two scenes sharing **A**, transitively. Components containing one of the 71 development anchors are assigned to `T_core`. The remaining components are sorted by SHA-256 of `seed:comma-separated-numerically-sorted-scene-ids`, then cyclically assigned to `T_core`, `T_head`, `V_select`, `T_test`. Components cannot be split to increase evaluation sample counts.

Within each side, scenes are sorted by SHA-256 of `seed:scene-id` and limited by its cap. Unselected scenes remain assigned to that side; they cannot migrate to another split. Development components are always training-side, though a hash cap may leave individual previously seen scenes unselected. The manifest saves all component membership and assignment, selected IDs, component counts and largest selected component size, every label member's SHA-256, archive SHA-256, portable archive basename, units, registration policy, and its own canonical-content SHA-256. `validate_manifest` rejects content hash mismatch and assignment inconsistencies. Build-time coverage is all labels only; selected input/node/edge coverage must be recorded during preprocessing, not invented from labels.

The original 53 development scenes are `0,1,2,100,200,…,4900,4999`. The additional 18 Level2 anchors share A identities with Level1 scene 0, previewed during intake; their provenance and exact IDs are in [Level1 intake](level1_intake.md). These 18 Level2 scenes were not themselves opened as measurement/reference data. Their entire connected component is nevertheless forced to training. A pre-intake manifest omitting these anchors is rejected by `validate_manifest_config`; create a new manifest/output rather than editing old results. Labels themselves are allowed for split construction; no measurements from unselected test scenes are inspected by manifest creation. A-B identity groups necessarily remain within the broader shared-A components.

## Explicit 64-token temporal registration

Policy name: **`scene64-quantized-endpoints-in-support-v1`**.

A common nominal scene grid runs from the earliest to latest valid measurement. For each track, its support start/end map to the floor/ceil nominal-grid positions; at least two distinct slots are reserved. Actual sample times are then spaced uniformly **within that track's actual support**, with both endpoints exact, and assigned to those contiguous slots. Lat/lon/velocity use linear interpolation. Course is unwrapped along the shortest angular arc before interpolation and wrapped to [0°,360°) afterward. No coordinate is extrapolated beyond observed support.

This is an explicit temporal quantization/time-warp approximation. `observed=True` means “this token contains a real or interpolated **in-support value**”; it **does not mean** the value occurred at the nominal slot's timestamp. `observation_times[N,L]` records actual sampling times, with NaN outside support slots. The nominal `grid[L]` is separate. Inputs outside the support slots are zero placeholders for embedding-space completion; the observed mask controls their use.

The policy retains short fragments that would otherwise have zero/one samples on a fixed 64-point scene grid. Each retained track has at least two true endpoint anchors, without extending its actual temporal support. The audit records per-track endpoints and occupied slots, number of nominal points inside actual support, maximum absolute actual-versus-nominal time displacement, and the number of short-support tracks. The model consumes quantized positions; actual times are diagnostic metadata, not an additional model feature in v1. All reference branches and correction arms share exactly this policy. It is a pragmatic implementation interpretation, **not an author-verified TMTAFF registration**.

The normalizer must fit only observed values in the applicable reference's training material (`T_core`, or `T_core ∪ T_head` for `A_full`), then remain frozen. Evaluation data never determines its moments. The resampler performs no normalization itself.
