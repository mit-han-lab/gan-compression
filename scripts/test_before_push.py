import os


def run(command):
    print(command)
    exit_status = os.system(command)
    if exit_status > 0:
        exit(1)


if __name__ == '__main__':
    # Download mini datasets
    if not os.path.exists('database/mini'):
        run('bash ./datasets/download_cyclegan_dataset.sh mini')
    if not os.path.exists('./database/mini_pix2pix'):
        run('bash ./datasets/download_cyclegan_dataset.sh mini_pix2pix')

    # Test get_real_stat.py
    if not os.path.exists('tmp/real_stat/mini.npz'):
        run('python get_real_stat.py --dataset_mode single '
            '--data_path database/mimi/valB '
            '--output_path tmp/real_stat/mini.npz')
    elif not os.path.exists('tmp/real_stat/mini_pix2pix.npz'):
        run('python get_real_stat.py '
            '--data_path database/mimi_pix2pix '
            '--output_path tmp/real_stat/mini_pix2pix.npz')

    # Test pretrained cycleGAN models
    if not os.path.exists('pretrained/cycle_gan/horse2zebra/full/latest_net_G.pth'):
        run('python scripts/download_model.py --stage full --model cycle_gan --task horse2zebra')
    run('python test.py --dataroot database/mini/valA '
        '--dataset_mode single '
        '--results_dir tmp/results '
        '--ngf 64 --netG resnet_9blocks '
        '--restore_G_path pretrained/cycle_gan/horse2zebra/full/latest_net_G.pth '
        '--need_profile --real_stat_path tmp/real_stat/mini.npz')

    if not os.path.exists('pretrained/cycle_gan/horse2zebra/distill/latest_net_G.pth'):
        run('python scripts/download_model.py --stage distill --model cycle_gan --task horse2zebra')
    run('python test.py --dataroot database/mini/valA '
        '--dataset_mode single '
        '--results_dir tmp/results '
        '--ngf 32 --netG mobile_resnet_9blocks '
        '--restore_G_path pretrained/cycle_gan/horse2zebra/distill/latest_net_G.pth '
        '--need_profile --real_stat_path tmp/real_stat/mini.npz')

    if not os.path.exists('pretrained/cycle_gan/horse2zebra/supernet/latest_net_G.pth'):
        run('python scripts/download_model.py --stage supernet --model cycle_gan --task horse2zebra')
    run('python test.py --dataroot database/mini/valA '
        '--dataset_mode single '
        '--results_dir tmp/results '
        '--ngf 32 --netG super_mobile_resnet_9blocks '
        '--config_str 16_16_32_16_32_32_16_16 '
        '--restore_G_path pretrained/cycle_gan/horse2zebra/supernet/latest_net_G.pth '
        '--need_profile --real_stat_path tmp/real_stat/mini.npz')

    if not os.path.exists('pretrained/cycle_gan/horse2zebra/compressed/latest_net_G.pth'):
        run('python scripts/download_model.py --stage compressed --model cycle_gan --task horse2zebra')
    run('python test.py --dataroot database/mini/valA '
        '--dataset_mode single '
        '--results_dir tmp/results '
        '--netG sub_mobile_resnet_9blocks '
        '--config_str 16_16_32_16_32_32_16_16 '
        '--restore_G_path pretrained/cycle_gan/horse2zebra/compressed/latest_net_G.pth '
        '--need_profile --real_stat_path tmp/real_stat/mini.npz')

    # Test pretrained pix2pix models
    if not os.path.exists('pretrained/pix2pix/edges2shoes-r/compressed/latest_net_G.pth'):
        run('python scripts/download_model.py --stage compressed --model pix2pix --task edges2shoes-r')
    run('python test.py --dataroot database/mini_pix2pix '
        '--dataset_mode aligned '
        '--results_dir tmp/results '
        '--netG sub_mobile_resnet_9blocks '
        '--config_str 32_32_48_32_48_48_16_16 '
        '--restore_G_path pretrained/pix2pix/edges2shoes-r/compressed/latest_net_G.pth '
        '--need_profile --real_stat_path tmp/real_stat/mini_pix2pix.npz')

    # Test cycleGAN train
    run('python train.py '
        '--dataroot database/mini '
        '--model cycle_gan --ngf 16 --ndf 16 --netG resnet_9blocks '
        '--log_dir tmp/logs/cycle_gan/train '
        '--real_stat_A_path tmp/real_stat/mini.npz '
        '--real_stat_B_path tmp/real_stat/mini.npz '
        '--nepochs 1 --nepochs_decay 0 '
        '--print_freq 1')

    # Test pix2pix train
    run('python train.py '
        '--dataroot database/mini_pix2pix '
        '--model pix2pix --ngf 16 --ndf 16 '
        '--log_dir tmp/logs/pix2pix/train '
        '--real_stat_path tmp/real_stat/mini_pix2pix.npz '
        '--nepochs 1 --nepochs_decay 0 '
        '--print_freq 1')

    # Test distillation
    run('python distill.py --dataroot database/mini_pix2pix '
        '--distiller resnet '
        '--log_dir tmp/logs/pix2pix/distill '
        '--student_ngf 8 --teacher_ngf 16 --ndf 16 '
        '--restore_teacher_G_path tmp/logs/pix2pix/train/checkpoints/latest_net_G.pth '
        '--restore_pretrained_G_path tmp/logs/pix2pix/train/checkpoints/latest_net_G.pth '
        '--restore_D_path tmp/logs/pix2pix/train/checkpoints/latest_net_D.pth '
        '--real_stat_path tmp/real_stat/mini_pix2pix.npz '
        '--nepochs 1 --nepochs_decay 0 --save_epoch_freq 20 '
        '--print_freq 1')

    # Test supernet training
    run('python train_supernet.py --dataroot database/mini_pix2pix '
        '--distiller resnet --config_set test '
        '--log_dir tmp/logs/pix2pix/supernet '
        '--student_ngf 8 --teacher_ngf 16 --ndf 16 '
        '--restore_teacher_G_path tmp/logs/pix2pix/train/checkpoints/latest_net_G.pth '
        '--restore_student_G_path tmp/logs/pix2pix/distill/checkpoints/latest_net_G.pth '
        '--restore_D_path tmp/logs/pix2pix/distill/checkpoints/latest_net_D.pth '
        '--real_stat_path tmp/real_stat/mini_pix2pix.npz '
        '--nepochs 1 --nepochs_decay 0 --save_epoch_freq 20 '
        '--print_freq 1')

    # Test fine-tuning
    run('python distill.py --dataroot database/mini_pix2pix '
        '--distiller resnet --config_str 8_6_6_8_8_8_8_8 '
        '--log_dir tmp/logs/pix2pix/finetune '
        '--student_ngf 8 --teacher_ngf 16 --ndf 16 '
        '--restore_teacher_G_path tmp/logs/pix2pix/train/checkpoints/latest_net_G.pth '
        '--restore_student_G_path tmp/logs/pix2pix/supernet/checkpoints/latest_net_G.pth '
        '--restore_D_path tmp/logs/pix2pix/supernet/checkpoints/latest_net_D.pth '
        '--real_stat_path tmp/real_stat/mini_pix2pix.npz '
        '--nepochs 1 --nepochs_decay 0 --save_epoch_freq 20 '
        '--print_freq 1')

    # Test export
    run('python export.py '
        '--input_path tmp/logs/pix2pix/finetune/checkpoints/latest_net_G.pth '
        '--output_path tmp/logs/pix2pix/compressed/checkpoints/compressed_net_G.pth '
        '--config_str 8_6_6_8_8_8_8_8')
