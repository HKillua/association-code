"""087 explicit temporal scope over the existing paired sequence trainer."""
from association.models.helper_temporal import SCOPE
from association.training import sequence_position as sequence


def configuration(value):
    result=sequence.configuration(value)
    if result['scope']!=SCOPE: raise ValueError('temporal scope required')
    return result


def load_checkpoint(path, *, device='cpu'):
    model,saved=sequence.load_checkpoint(path,device=device)
    configuration(saved['configuration'])
    return model,saved


def train_helper_temporal(train_raw, config, output, *, device='cpu', check=None, paired_reference_fit=None):
    configuration(config)
    return sequence.train_sequence_position(train_raw,config,output,device=device,check=check,paired_reference_fit=paired_reference_fit)
