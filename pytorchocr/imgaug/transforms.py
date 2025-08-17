from pytorchocr.imgaug.rec_aug import RecAug

TRANSFORM_DICTS = {
    "RecAug" : RecAug,
}

def create_transforms(transform_param_list, global_config=None):
    transform_funcs = []
    for transform in transform_param_list:
        assert isinstance(transform, dict) and len(transform) == 1, "yaml format is not expected. Double check your Transform"

        transform_name = list(transform)[0]
        assert transform_name in TRANSFORM_DICTS, "{} is not registered in transformer dict.".format(transform_name)

        param = {} if transform[transform_name] is None else transform[transform_name]
        if global_config is not None:
            param.update(global_config)

        transform_fn = TRANSFORM_DICTS[transform_name]
        transform_funcs.append(transform_fn)
    return transform_funcs
