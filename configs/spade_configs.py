import random


class SPADEConfigs:
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

    def all_configs(self):

        def yield_channels(i):
            if i == len(self.n_channels):
                yield []
                return
            for n in self.n_channels[i]:
                for after_channels in yield_channels(i + 1):
                    yield [n] + after_channels

        for i, channels in enumerate(yield_channels(0)):
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


def get_configs(config_name):
    if config_name == 'channels-48':
        return SPADEConfigs(n_channels=[[48, 40, 32],
                                        [48, 40, 32], [48, 40, 32], [48, 40, 32], [48, 40, 32],
                                        [48, 40, 32, 24], [48, 40, 32, 24], [48, 40, 32, 24]])
    elif config_name == 'channels-48-stage1':
        return SPADEConfigs(n_channels=[[48, 40],
                                        [48, 40], [48, 40], [48, 40], [48, 40],
                                        [48, 40], [48, 40], [48, 40]])
    elif config_name == 'channels-64-stage1':
        return SPADEConfigs(n_channels=[[64, 48],
                                        [64, 48], [64, 48], [64, 48], [64, 48],
                                        [64, 48, 40], [64, 48, 40], [64, 48, 40]])
    elif config_name == 'channels-64':
        return SPADEConfigs(n_channels=[[64, 48, 40, 32],
                                        [64, 48, 40, 32], [64, 48, 40, 32], [64, 48, 40, 32], [64, 48, 40, 32],
                                        [64, 48, 40, 32, 24], [64, 48, 40, 32, 24], [64, 48, 40, 32, 24]])
    elif config_name == 'channels-64-coco':
        return SPADEConfigs(n_channels=[[64, 48, 40, 32, 24], [64, 48, 40, 32, 24],
                                        [64, 48, 40, 32, 24], [64, 48, 40, 32, 24], [64, 48, 40, 32, 24],
                                        [64, 48, 40, 32, 24], [64, 48, 40, 32, 24], [64, 48, 40, 32, 24]])
    elif config_name == 'smallest':
        return SPADEConfigs(n_channels=[[24],
                                        [24], [24], [24], [24],
                                        [24], [24], [24]])
    elif config_name == 'middle':
        return SPADEConfigs(n_channels=[[48],
                                        [48], [48], [48], [48],
                                        [40], [40], [40]])
    elif config_name == 'debug':
        return SPADEConfigs(n_channels=[[48],
                                        [48], [48], [48], [48],
                                        [48], [48], [48]])

    raise NotImplementedError('Unknown configuration [%s]!!!' % config_name)
