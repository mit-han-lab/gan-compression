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
        self.use_wandb = opt.use_wandb
        if self.use_wandb: 
            import wandb
            wandb.init(config=self.opt, project="fomm-compression")
#             wandb.watch(model)
            # manually write opts to wandb
#             wandb.config.model = opt.model
#             wandb.config.dataset_mode = opt.dataset_mode
#             wandb.config.log_dir = opt.log_dir
#             wandb.config.real_stat_path = opt.real_stat_path
#             wandb.config.batch_size = opt.batch_size
#             wandb.config.ngf = opt.ngf
#             wandb.config.nepochs = opt.nepochs
#             wandb.config.nepochs_decay = opt.nepochs_decay
#             wandb.config.save_epoch_freq = opt.save_epoch_freq
#             wandb.config.save_latest_freq = opt.save_latest_freq
#             wandb.config.eval_batch_size = opt.eval_batch_size
#             wandb.config.num_threads = opt.num_threads
#             wandb.config.use_coord = opt.use_coord
#             wandb.config.use_motion = opt.use_motion

    def plot(self, items, step):
        if len(items) == 0:
            return
        for k, v in items.items():
            self.writer.add_scalar(k, v, global_step=step)
            if self.use_wandb:
                wandb.log({f'{k}': v}, step=step)
        self.writer.flush()

    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)

        for k, v in errors.items():
            if 'Specific' in k:
                continue
            kk = k.split('/')[-1]
            message += '%s: %.3f ' % (kk, v)
#             if self.use_wandb:
#                 wandb.log({f"epoch": epoch, f"iters": i, f"time": t, f"{kk}": v}, step=i)

        print(message, flush=True)
        self.log_file.write('%s\n' % message)
        self.log_file.flush()
        

    def print_current_metrics(self, epoch, i, metrics, t):
        message = '###(Evaluate epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)

        for k, v in metrics.items():
            kk = k.split('/')[-1]
            message += '%s: %.3f ' % (kk, v)
#             if self.use_wandb:
#                 wandb.log({f"epoch": epoch, f"iters": i, f"time": t, f"{kk}": v}, step=i)

        print(message, flush=True)
        self.log_file.write('%s\n' % message)

    def print_info(self, message):
        print(message, flush=True)
        self.log_file.write(message + '\n')
