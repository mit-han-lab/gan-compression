import random


class ChannelConfigs:
    def __init__(self, n_channels, weights=None):
        self.attributes = ['n_channels']
        self.n_channels = n_channels
        self.weights = weights

    def sample(self, weighted_sample=1, weight_strategy='arithmetic'):
        ret = {}
        ret['channels'] = []
        for n_channel in self.n_channels:
            if weighted_sample > 1.0001:
                weights = []
                if weight_strategy == 'arithmetic':
                    now = (len(n_channel) - 1) / (weighted_sample - 1)
                    while len(weights) < len(n_channel):
                        weights.append(now)
                        now += 1
                elif weight_strategy == 'geometric':
                    now = 1
                    while len(weights) < len(n_channel):
                        weights.append(now)
                        now *= weighted_sample
                else:
                    raise NotImplementedError('Unknown weight strategy [%s]!!!' % weight_strategy)
            else:
                weights = None
            ret['channels'].append(random.choices(n_channel, weights=weights)[0])
        return ret

    def sample_layer(self, layer):
        return random.choice(self.n_channels[layer])

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

    def __len__(self):
        ret = 1
        for n_channel in self.n_channels:
            ret *= len(n_channel)
