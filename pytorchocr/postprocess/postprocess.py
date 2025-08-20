from .rec_postprocess import CTCLabelDecode

POSTPROCESS_DICT = {
    "CTCLabelDecode" : CTCLabelDecode,
}

def build_postprocess(config, global_config = None):
    assert 'name' in config, "name should be set in config['PostProcess']"
    assert config['name'] in POSTPROCESS_DICT, "{} is not registered in POSTPROCESS_DICT".format(config['name'])

    if global_config is not None:
        config.update(global_config)

    return POSTPROCESS_DICT[config['name']](**config)
