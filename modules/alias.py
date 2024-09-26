"""
Getting model by alias.
"""

from .models import (
    LongShortTermMemory,
    CNN1D,
    CNNLSTM,
    TransformerTS
)


def get_by_alias(alias: str, **kwargs):
    """
    Get model by alias. This function will take alias and return the model.
    """
    def filter_and_format(alias: str, kws: dict):
        # Filter out the keys that are not in the model's parameters
        return {k.split('_', 1)[1]: v for k, v in kws.items() if alias in k}

    
    if alias == 'lstm':
        return LongShortTermMemory(**filter_and_format('lstm', kwargs))
    if alias == 'cnn1d':
        return CNN1D(**filter_and_format('cnn1d', kwargs))
    if alias == 'cnnlstm':
        return CNNLSTM(**filter_and_format('cnnlstm', kwargs))
    if alias == 'transformer':
        return TransformerTS(**filter_and_format('transformer', kwargs))
    raise ValueError(
        f'Invalid model alias. Got: {alias}')


def get_by_aliases(aliases: list[str], **kwargs):
    """
    Get models by aliases.
    """
    return [get_by_alias(alias, **kwargs) for alias in aliases]
