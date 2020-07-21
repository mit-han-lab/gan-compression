import random


class ResnetConfigs:
    def __init__(self, n_channels):
        self.attributes = ['n_channels']
        self.n_channels = n_channels

    def sample(self):
        ret = {}
        ret['channels'] = []
        for n_channel in self.n_channels:
            ret['channels'].append(random.choice(n_channel))
        return ret

    def largest(self):
        ret = {}
        ret['channels'] = []
        for n_channel in self.n_channels:
            ret['channels'].append(max(n_channel))
        return ret

    def smallest(self):
        ret = {}
        ret['channels'] = []
        for n_channel in self.n_channels:
            ret['channels'].append(min(n_channel))
        return ret

    def all_configs(self, split=1, remainder=0):

        def yield_channels(i):
            if i == len(self.n_channels):
                yield []
                return
            for n in self.n_channels[i]:
                for after_channels in yield_channels(i + 1):
                    yield [n] + after_channels

        for i, channels in enumerate(yield_channels(0)):
            if i % split == remainder:
                yield {'channels': channels}

    def __call__(self, name):
        assert name in ('largest', 'smallest')
        if name == 'largest':
            return self.largest()
        elif name == 'smallest':
            return self.smallest()
        else:
            raise NotImplementedError

    def __str__(self):
        ret = ''
        for attr in self.attributes:
            ret += 'attr: %s\n' % str(getattr(self, attr))
        return ret

    def __len__(self):
        ret = 1
        for n_channel in self.n_channels:
            ret *= len(n_channel)


def get_configs(config_name):
    if config_name == 'channels-48':
        return ResnetConfigs(n_channels=[[48, 32], [48, 32], [48, 40, 32],
                                         [48, 40, 32], [48, 40, 32], [48, 40, 32],
                                         [48, 32, 24, 16], [48, 32, 24, 16]])
    elif config_name == 'channels-32':
        return ResnetConfigs(n_channels=[[32, 24, 16], [32, 24, 16], [32, 24, 16],
                                         [32, 24, 16], [32, 24, 16], [32, 24, 16],
                                         [32, 24, 16], [32, 24, 16]])
    elif config_name == 'channels-64-cycleGAN-stage1':
        return ResnetConfigs(n_channels=[[64, 48, 32], [64, 48, 32], [64, 48, 32],
                                         [64, 48, 32], [64, 48, 32], [64, 48, 32],
                                         [64, 48, 32], [64, 48, 32]])
    elif config_name == 'channels-64-cycleGAN':
        return ResnetConfigs(n_channels=[[64, 48, 32, 24, 16], [64, 48, 32, 24, 16], [64, 48, 32, 24, 16],
                                         [64, 48, 32, 24, 16], [64, 48, 32, 24, 16], [64, 48, 32, 24, 16],
                                         [64, 48, 32, 24, 16], [64, 48, 32, 24, 16]])
    elif config_name == 'channels-64-pix2pix-stage1':
        return ResnetConfigs(n_channels=[[64, 48], [64, 48], [64, 48],
                                         [64, 48], [64, 48], [64, 48],
                                         [64, 48, 32], [64, 48, 32]])
    elif config_name == 'channels-64-pix2pix':
        return ResnetConfigs(n_channels=[[64, 48, 32], [64, 48, 32], [64, 48, 40, 32],
                                         [64, 48, 40, 32], [64, 48, 40, 32], [64, 48, 40, 32],
                                         [64, 48, 32, 24, 16], [64, 48, 32, 24, 16]])
    elif config_name == 'debug':
        return ResnetConfigs(n_channels=[[48, 32], [48, 32], [48, 40, 32],
                                         [48, 40, 32], [48, 40, 32], [48, 40, 32],
                                         [48], [48]])
    elif config_name == 'test':
        return ResnetConfigs(n_channels=[[8], [6, 8], [6, 8],
                                         [8], [8], [8],
                                         [8], [8]])
    else:
        raise NotImplementedError('Unknown configuration [%s]!!!' % config_name)
