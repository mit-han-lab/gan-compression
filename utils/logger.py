import os
import time

from tensorboardX import SummaryWriter


class Logger:
    def __init__(self, opt):
        self.opt = opt
        self.log_file = open(os.path.join(opt.log_dir, 'log.txt'), 'a')
        os.makedirs(opt.tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(opt.tensorboard_dir)
        now = time.strftime('%c')
        self.log_file.write('================ (%s) ================\n' % now)
        self.log_file.flush()

    def plot(self, items, step):
        if len(items) == 0:
            return
        for k, v in items.items():
            self.writer.add_scalar(k, v, global_step=step)
        self.writer.flush()

    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)

        for k, v in errors.items():
            if 'Specific' in k:
                continue
            kk = k.split('/')[-1]
            message += '%s: %.3f ' % (kk, v)

        print(message, flush=True)
        self.log_file.write('%s\n' % message)
        self.log_file.flush()

    def print_current_metrics(self, epoch, i, metrics, t):
        message = '###(Evaluate epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)

        for k, v in metrics.items():
            kk = k.split('/')[-1]
            message += '%s: %.3f ' % (kk, v)

        print(message, flush=True)
        self.log_file.write('%s\n' % message)

    def print_info(self, message):
        print(message, flush=True)
        self.log_file.write(message + '\n')
