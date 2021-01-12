from configs.channel_configs import ChannelConfigs


def get_configs(config_name):
    if config_name == 'channels-64':
        return ChannelConfigs(n_channels=[[64, 56, 48, 40, 32, 24, 16], [64, 56, 48, 40, 32, 24, 16],
                                          [64, 56, 48, 40, 32, 24, 16], [64, 56, 48, 40, 32, 24, 16],
                                          [64, 56, 48, 40, 32, 24, 16], [64, 56, 48, 40, 32, 24, 16],
                                          [64, 56, 48, 40, 32, 24, 16], [64, 56, 48, 40, 32, 24, 16],
                                          [64, 56, 48, 40, 32, 24, 16], [64, 56, 48, 40, 32, 24, 16]])
    else:
        raise NotImplementedError('Unknown configuration [%s]!!!' % config_name)
