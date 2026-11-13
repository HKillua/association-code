"""Fresh Level1 scene selection from labels only; measurements remain unopened."""
from __future__ import annotations

from collections import Counter, defaultdict

from association.data.archive import (
    DataError, SPLIT_NAMES, UNIT_CONTRACT, _digest, _numeric_sort, _scene_id,
    validate_manifest,
)


def validate_fresh_scene_usage(manifest):
    """Validate explicit preview exposure without inventing prior split roles."""
    if manifest.get('dataset_level') != 'Level1':
        raise DataError('Fresh Level1 scene selection must declare dataset_level Level1')
    development = manifest['development_scenes']
    if (len(development) != len(set(development))
            or not set(development).issubset(manifest['splits']['T_core'])):
        raise DataError('Previewed development scenes must remain in T_core')
    expected = dict.fromkeys(SPLIT_NAMES, 0)
    expected['T_core'] = len(development)
    if manifest['historical_protection_counts'] != expected:
        raise DataError('Preview protection counts differ from actual scenes')
    if manifest['historical_usage'] != {
        'kind': 'preview_only', 'previewed_scenes': development,
        'prior_measurement_splits': None,
    }:
        raise DataError('Fresh scene selection must state its preview-only exposure')


def build_fresh_scene_manifest(archive, seed=20, caps=None, *, development_scenes=('0',)):
    """Freeze a scene split while preserving real shared-A dependence metadata.

    All label CSVs are opened to construct shared-A components. No measurement or
    reference CSV is opened, including for the selected sealed test scenes.
    The SHA ordering is the existing scene-selection algorithm, not a file hash.
    """
    if archive.dataset_level != 'Level1':
        raise DataError('Fresh Level1 selection requires Level1 member directories; external ZIP names are not proof')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise DataError('Scene selection seed must be an integer')
    caps = dict(caps if caps is not None else dict(T_core=2100, T_head=0, V_select=300, T_test=600))
    if (set(caps) != set(SPLIT_NAMES) or caps['T_head'] != 0
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in caps.values())):
        raise DataError('Scene protocol requires nonnegative caps and an empty T_head')
    scene_ids = archive.scene_ids()
    development = {_scene_id(sid) for sid in development_scenes}
    if not development.issubset(scene_ids):
        raise DataError('Development scene IDs absent from archive')
    if caps['T_core'] < len(development) or sum(caps.values()) > len(scene_ids):
        raise DataError('Requested counts cannot retain previewed scenes or exceed archive inventory')

    parent = dict(zip(scene_ids, scene_ids))

    def find(sid):
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]
            sid = parent[sid]
        return sid

    ancestor_first, ancestors_by_scene = {}, {}
    for sid in scene_ids:
        ancestors = {row['mmsi'].split('-', 1)[0] for row in archive.read_labels(sid)}
        archive._label_cache.pop(sid, None)
        ancestors_by_scene[sid] = ancestors
        for ancestor in ancestors:
            if ancestor in ancestor_first:
                left, right = find(sid), find(ancestor_first[ancestor])
                if left != right:
                    parent[max(left, right, key=int)] = min(left, right, key=int)
            else:
                ancestor_first[ancestor] = sid
    grouped = defaultdict(list)
    for sid in scene_ids:
        grouped[find(sid)].append(sid)
    component_members = {'component-' + _digest(','.join(members))[:24]: members
                         for members in grouped.values()}
    components = {sid: cid for cid, members in component_members.items() for sid in members}
    free = sorted(set(scene_ids) - development,
                  key=lambda sid: (_digest(str(seed) + ':' + sid), int(sid)))
    splits, offset = {}, 0
    for split in SPLIT_NAMES:
        protected = development if split == 'T_core' else set()
        stop = offset + caps[split] - len(protected)
        splits[split] = _numeric_sort(protected | set(free[offset:stop]))
        offset = stop
    scene_splits = dict.fromkeys(scene_ids, 'unused')
    scene_splits.update({sid: split for split, scenes in splits.items() for sid in scenes})
    ancestors_by_split = {split: set().union(*(ancestors_by_scene[sid] for sid in scenes))
                          for split, scenes in splits.items()}
    active_splits = ('T_core', 'V_select', 'T_test')
    cross_split_ancestors = {
        left + '__' + right: len(ancestors_by_split[left] & ancestors_by_split[right])
        for index, left in enumerate(active_splits) for right in active_splits[index + 1:]
    }
    development = _numeric_sort(development)
    manifest = dict(
        manifest_version=3, protocol='MTAD-Level1 fresh scene-level v1',
        dataset_level='Level1', split_origin='fresh_archive', split_unit='scene', seed=seed,
        caps=caps, splits=splits, all_scenes=scene_ids, scene_splits=scene_splits,
        components=components, component_members=component_members,
        development_scenes=development, label_members=dict(archive.label_members),
        archive=archive.metadata, unit_contract=UNIT_CONTRACT.copy(),
        historical_usage=dict(kind='preview_only', previewed_scenes=development,
                              prior_measurement_splits=None),
        historical_protection_counts={split: len(development) if split == 'T_core' else 0
                                      for split in SPLIT_NAMES},
        split_counts={split: dict(selected_scenes=len(scenes),
            selected_components=len({components[sid] for sid in scenes}),
            largest_selected_component_scenes=max(Counter(components[sid] for sid in scenes).values(), default=0),
            unique_ancestors=len(ancestors_by_split[split])) for split, scenes in splits.items()},
        dependence=dict(shared_A_counts=cross_split_ancestors,
            selected_components_crossing_splits=sum(
                len({scene_splits[sid] for sid in members} - {'unused'}) > 1
                for members in component_members.values()),
            identity_independent=False,
            prior_other_level_development='Level2 experiments used overlapping source identities; '
                'this scene split is not an identity-independent holdout from that development.'),
        split_policy='scene-level; preview scenes T_core; remaining IDs sorted by '
            "SHA256(str(seed)+':'+scene_id), fill T_core then V_select then T_test; remainder unused",
        selection_note="Project scene list, not the authors' published list. Seed is our split choice. "
            'Shared-A components remain for dependence reporting and may cross splits. '
            'The digest only orders IDs and is not an admission gate.',
        content_coverage='all labels and ZIP directory; zero measurement or reference reads; '
            'T_test measurement/reference remains sealed',
        preparation_status='manifest only; Level1 time alignment, training statistics and training pending',
    )
    validate_manifest(manifest)
    return manifest
