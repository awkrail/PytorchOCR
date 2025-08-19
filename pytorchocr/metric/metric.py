from .rec_metric import RecMetric

METRIC_DICT = {
    'RecMetric' : RecMetric,
}

def build_metric(config):
    assert 'name' in config, "name should be set in metric_config"
    assert config['name'] in METRIC_DICT, "{} does not exist in METRIC_DICT".format(config['name'])
    metric = METRIC_DICT[config['name']](**config)
    return metric
