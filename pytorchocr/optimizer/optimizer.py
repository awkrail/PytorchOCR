import torch.optim as optim 

SCHEDULER_DICT = {
    "Cosine" : optim.lr_scheduler.CosineAnnealingLR,
}

OPTIM_DICT = {
    "Adam" : optim.Adam,
}

def juild_lr_scheduler(lr_config, epochs, step_each_epochs):
    assert 'name' in lr_config, "name should be set in lr_config"
    assert lr_config['name'] in SCHEDULER_DICT, "{} does not exist in SCHEDULER_DICT".format(lr_config['name'])
    scheduler = SCHEDULER_DICT[lr_config['name']]()
    return scheduler

def build_optimizer(config, epochs, step_each_epoch):
    assert 'name' in config and 'lr' in config, "name and lr should be set in optimizer config"
    assert config['name'] in OPTIM_DICT, "{} does not exist in OPTIM_DICT".format(config['name'])
    assert 'name' in config['lr'], "name should be set in lr_config"
    assert config['lr']['name'] in SCHEDULER_DICT, "{} does not exist in SCHEDULER_DICT".format(lr_config['name'])

    lr_config = config['lr']
    lr = lr_config.get('learning_rate', 1e-3)
    weight_decay = lr_config.get('weight_decay', 1e-5)

    optimizer = OPTIM_DICT[config['name']](model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = SCHEDULER_DICT[lr_config['name']](optimizer, T_max=epochs)
    return optimizer, scheduler
