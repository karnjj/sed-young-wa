from pathlib import Path
if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom


@custom
def transform_custom(*args, **kwargs):
    CONFIG_DIR = kwargs['CONFIG_DIR']
    DATA_DIR = kwargs['DATA_DIR']

    Path(CONFIG_DIR).mkdir(parents=True, exist_ok=True)
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


