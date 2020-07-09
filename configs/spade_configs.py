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

        for channels in yield_channels(0):
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
    elif config_name == 'channels-48-part1':
        return SPADEConfigs(n_channels=[[48, 40, 32],
                                        [48, 40, 32], [48, 40, 32], [48, 40, 32], [48, 40, 32],
                                        [48, 24], [48, 24], [48, 24]])
    elif config_name == 'channels-48-part2':
        return SPADEConfigs(n_channels=[[48, 40, 32],
                                        [48, 40, 32], [48, 40, 32], [48, 40, 32], [48, 40, 32],
                                        [48, 24], [48, 24], [40, 32]])
    elif config_name == 'channels-48-part3':
        return SPADEConfigs(n_channels=[[48, 40, 32],
                                        [48, 40, 32], [48, 40, 32], [48, 40, 32], [48, 40, 32],
                                        [48, 24], [40, 32], [48, 24]])
    elif config_name == 'channels-48-part4':
        return SPADEConfigs(n_channels=[[48, 40, 32],
                                        [48, 40, 32], [48, 40, 32], [48, 40, 32], [48, 40, 32],
                                        [48, 24], [40, 32], [40, 32]])
    elif config_name == 'channels-48-part5':
        return SPADEConfigs(n_channels=[[48, 40, 32],
                                        [48, 40, 32], [48, 40, 32], [48, 40, 32], [48, 40, 32],
                                        [40, 32], [48, 24], [48, 24]])
    elif config_name == 'channels-48-part6':
        return SPADEConfigs(n_channels=[[48, 40, 32],
                                        [48, 40, 32], [48, 40, 32], [48, 40, 32], [48, 40, 32],
                                        [40, 32], [48, 24], [40, 32]])
    elif config_name == 'channels-48-part7':
        return SPADEConfigs(n_channels=[[48, 40, 32],
                                        [48, 40, 32], [48, 40, 32], [48, 40, 32], [48, 40, 32],
                                        [40, 32], [40, 32], [48, 24]])
    elif config_name == 'channels-48-part8':
        return SPADEConfigs(n_channels=[[48, 40, 32],
                                        [48, 40, 32], [48, 40, 32], [48, 40, 32], [48, 40, 32],
                                        [40, 32], [40, 32], [40, 32]])
    elif config_name == 'channels2-48':
        return SPADEConfigs(n_channels=[[48, 40, 32, 24],
                                        [48, 40, 32, 24], [48, 40, 32, 24], [48, 40, 32, 24], [48, 40, 32, 24],
                                        [48, 40, 32, 24], [48, 40, 32, 24], [48, 40, 32, 24]])
    elif config_name == 'channels2-48-part1':
        return SPADEConfigs(n_channels=[[48],
                                        [48, 24], [48, 40, 32, 24], [48, 40, 32, 24], [48, 40, 32, 24],
                                        [48, 40, 32, 24], [48, 40, 32, 24], [48, 40, 32, 24]])
    elif config_name == 'channels2-48-part2':
        return SPADEConfigs(n_channels=[[40],
                                        [48, 24], [48, 40, 32, 24], [48, 40, 32, 24], [48, 40, 32, 24],
                                        [48, 40, 32, 24], [48, 40, 32, 24], [48, 40, 32, 24]])
    elif config_name == 'channels2-48-part3':
        return SPADEConfigs(n_channels=[[32],
                                        [48, 24], [48, 40, 32, 24], [48, 40, 32, 24], [48, 40, 32, 24],
                                        [48, 40, 32, 24], [48, 40, 32, 24], [48, 40, 32, 24]])
    elif config_name == 'channels2-48-part4':
        return SPADEConfigs(n_channels=[[24],
                                        [48, 24], [48, 40, 32, 24], [48, 40, 32, 24], [48, 40, 32, 24],
                                        [48, 40, 32, 24], [48, 40, 32, 24], [48, 40, 32, 24]])
    elif config_name == 'channels2-48-part5':
        return SPADEConfigs(n_channels=[[48],
                                        [40, 32], [48, 40, 32, 24], [48, 40, 32, 24], [48, 40, 32, 24],
                                        [48, 40, 32, 24], [48, 40, 32, 24], [48, 40, 32, 24]])
    elif config_name == 'channels2-48-part6':
        return SPADEConfigs(n_channels=[[40],
                                        [40, 32], [48, 40, 32, 24], [48, 40, 32, 24], [48, 40, 32, 24],
                                        [48, 40, 32, 24], [48, 40, 32, 24], [48, 40, 32, 24]])
    elif config_name == 'channels2-48-part7':
        return SPADEConfigs(n_channels=[[32],
                                        [40, 32], [48, 40, 32, 24], [48, 40, 32, 24], [48, 40, 32, 24],
                                        [48, 40, 32, 24], [48, 40, 32, 24], [48, 40, 32, 24]])
    elif config_name == 'channels2-48-part8':
        return SPADEConfigs(n_channels=[[24],
                                        [40, 32], [48, 40, 32, 24], [48, 40, 32, 24], [48, 40, 32, 24],
                                        [48, 40, 32, 24], [48, 40, 32, 24], [48, 40, 32, 24]])
    elif config_name == 'debug':
        return SPADEConfigs(n_channels=[[48],
                                        [48], [48], [48], [48],
                                        [48], [48], [48]])
    raise NotImplementedError('Unknown configuration [%s]!!!' % config_name)
