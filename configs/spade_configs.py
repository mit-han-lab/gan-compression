from configs.channel_configs import ChannelConfigs


def get_configs(config_name):
    if config_name == 'channels-48':
        return ChannelConfigs(n_channels=[[48, 40, 32],
                                          [48, 40, 32], [48, 40, 32], [48, 40, 32], [48, 40, 32],
                                          [48, 40, 32, 24], [48, 40, 32, 24], [48, 40, 32, 24]])
    elif config_name == 'channels-64':
        return ChannelConfigs(n_channels=[[64, 48, 40, 32],
                                          [64, 48, 40, 32], [64, 48, 40, 32], [64, 48, 40, 32], [64, 48, 40, 32],
                                          [64, 48, 40, 32, 24], [64, 48, 40, 32, 24], [64, 48, 40, 32, 24]])
    elif config_name == 'channels-64-coco':
        return ChannelConfigs(n_channels=[[64, 48, 40, 32, 24], [64, 48, 40, 32, 24],
                                          [64, 48, 40, 32, 24], [64, 48, 40, 32, 24], [64, 48, 40, 32, 24],
                                          [64, 48, 40, 32, 24], [64, 48, 40, 32, 24], [64, 48, 40, 32, 24]])
    raise NotImplementedError('Unknown configuration [%s]!!!' % config_name)
