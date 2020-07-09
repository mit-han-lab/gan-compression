from torch.nn import functional as F

from models.modules.spade_modules.base_spade_distiller_modules import BaseSPADEDistillerModules
from models.modules.super_modules import SuperConv2d


class SPADESupernetModules(BaseSPADEDistillerModules):
    def __init__(self, opt):
        super(SPADESupernetModules, self).__init__(opt)

    def profile(self, input_semantics, config=None):
        raise NotImplementedError('The distiller is only for training!!!')

    def calc_distill_loss(self, Tacts, Sacts):
        losses = {}
        for i, netA in enumerate(self.netAs):
            assert isinstance(netA, SuperConv2d)
            layer = self.mapping_layers[i]
            Tact, Sact = Tacts[layer], Sacts[layer]
            Sact = netA(Sact, {'channel': netA.out_channels})
            loss = F.mse_loss(Sact, Tact)
            losses['G_distill%d' % i] = loss
        return sum(losses.values()) * self.opt.lambda_distill, losses

    def load_networks(self, verbose=True):
        super(SPADESupernetModules, self).load_networks(verbose)
