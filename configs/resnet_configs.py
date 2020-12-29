from configs.channel_configs import ChannelConfigs


def get_configs(config_name):
    if config_name == 'channels-48':
        return ChannelConfigs(n_channels=[[48, 32], [48, 32], [48, 40, 32],
                                          [48, 40, 32], [48, 40, 32], [48, 40, 32],
                                          [48, 32, 24, 16], [48, 32, 24, 16]])
    elif config_name == 'channels-32':
        return ChannelConfigs(n_channels=[[32, 24, 16], [32, 24, 16], [32, 24, 16],
                                          [32, 24, 16], [32, 24, 16], [32, 24, 16],
                                          [32, 24, 16], [32, 24, 16]])
    elif config_name == 'channels-64-cycleGAN':
        return ChannelConfigs(n_channels=[[64, 48, 32, 24, 16], [64, 48, 32, 24, 16], [64, 48, 32, 24, 16],
                                          [64, 48, 32, 24, 16], [64, 48, 32, 24, 16], [64, 48, 32, 24, 16],
                                          [64, 48, 32, 24, 16], [64, 48, 32, 24, 16]])
    elif config_name == 'channels-64-pix2pix':
        return ChannelConfigs(
            n_channels=[[64, 56, 48, 40, 32, 24, 16], [64, 56, 48, 40, 32, 24, 16], [64, 56, 48, 40, 32, 24, 16],
                        [64, 56, 48, 40, 32, 24, 16], [64, 56, 48, 40, 32, 24, 16], [64, 56, 48, 40, 32, 24, 16],
                        [64, 56, 48, 40, 32, 24, 16], [64, 56, 48, 40, 32, 24, 16]])
    elif config_name == 'test':
        return ChannelConfigs(n_channels=[[8], [6, 8], [6, 8],
                                          [8], [8], [8],
                                          [8], [8]])
    else:
        raise NotImplementedError('Unknown configuration [%s]!!!' % config_name)
