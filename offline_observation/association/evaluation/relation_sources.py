"""Read-only binding of completed 098 endpoints for current peer comparisons.

The closed 098 training/analysis CLI is not needed to inspect its saved outputs.
These configuration and population checks retain the original experiment rules.
"""
from pathlib import Path

from association.training.query_relation import configuration
from association.utils.runtime import read_json

ROOT = Path(__file__).resolve().parents[2]

CONFIGS = {
    (20261023, 'generic'): 'query-relation-generic-level1.json',
    (20261023, 'relation'): 'query-relation-biased-level1.json',
    (20261021, 'generic'): 'query-relation-generic-replica-level1.json',
    (20261021, 'relation'): 'query-relation-biased-replica-level1.json',
}


def frozen_configuration(config):
    configuration(config)
    key = (config['training']['seed'], config['model']['mode'])
    if key not in CONFIGS or config != read_json(ROOT/'configs'/CONFIGS[key]):
        raise ValueError('098 requires one of four frozen seed/mode configurations')
    return config


def declared_manifest():
    return read_json(ROOT/'configs/data/mtad-level1-scene3000.json')


def relation_metadata(path):
    path = Path(path).resolve()
    run, summary, fit = [read_json(path/name) for name in ('run.json', 'summary.json', 'fit/run.json')]
    config = frozen_configuration(run['configuration'])
    mode, steps = config['model']['mode'], config['training']['updates']
    labels = ['initial', 'final'] + (['relation_off'] if mode == 'relation' else [])
    for record in (run, summary, fit):
        if (record.get('status') != 'COMPLETED' or record.get('configuration') != config
                or record.get('optimizer_updates') != steps or record.get('test_opened') is not False):
            raise ValueError('098 completed configuration/update chain mismatch')
    if (run.get('completed_endpoints') != labels or run.get('validation_selection') is not False
            or summary.get('source_system') != run.get('source_system')
            or summary.get('paired_control') != run.get('paired_control')
            or summary.get('budget_control') != run.get('budget_control')
            or not run.get('paired_control') or not run.get('budget_control')
            or fit.get('final_checkpoint') != 'final.pt'):
        raise ValueError('098 complete endpoint/source/budget provenance mismatch')
    manifest = declared_manifest()
    if fit.get('training_scene_ids') != manifest['splits']['T_core']:
        raise ValueError('098 complete training population differs')
    endpoints = {}
    for label in labels:
        step = 0 if label == 'initial' else steps
        filename = 'step-0.pt' if label == 'initial' else 'final.pt'
        enabled = mode == 'relation' and label != 'relation_off'
        endpoint = read_json(path/label/'summary.json')
        for record in (read_json(path/label/'run.json'), endpoint):
            provenance = record.get('checkpoint', {})
            if (record.get('status') != 'COMPLETED' or record.get('configuration') != config
                    or record.get('source_system') != run['source_system'] or record.get('device') != 'cpu'
                    or record.get('optimizer_updates') != 0 or record.get('test_opened') is not False
                    or record.get('groups_recomputed') is not False or record.get('threshold_reselected') is not False
                    or record.get('initial') != (label == 'initial')
                    or record.get('relation_enabled') is not enabled
                    or provenance.get('step') != step or provenance.get('actual_updates') != step
                    or provenance.get('configured_final_updates') != steps
                    or Path(provenance.get('path', '')).resolve() != path/'fit'/filename):
                raise ValueError('098 endpoint does not bind its actual checkpoint/bias state')
        if endpoint['scene_ids'] != manifest['splits']['V_select']:
            raise ValueError('098 full V population/order mismatch')
        components = {sid: str(cid) for cid, item in endpoint['components'].items() for sid in item['scene_ids']}
        if components != {sid: str(manifest['components'][sid]) for sid in endpoint['scene_ids']}:
            raise ValueError('098 component registry mismatch')
        for directory in ('predictions', 'system-predictions'):
            if {p.name for p in (path/label/directory).glob('*.npz')} != {f'states-{sid}.npz' for sid in endpoint['scene_ids']}:
                raise ValueError('098 endpoint cache inventory mismatch')
        endpoints[label] = endpoint
    return mode, dict(path=path, configuration=config, method='learned_sequence',
        source_system=run['source_system'], scene_ids=endpoint['scene_ids'], components=components,
        summary=endpoints['final'], endpoints=endpoints,
        paired_control=Path(run['paired_control']).resolve(), budget_control=Path(run['budget_control']).resolve())
