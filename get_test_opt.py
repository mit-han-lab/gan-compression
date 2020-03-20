import os
import pickle

from options.test_options import TestOptions

if __name__ == '__main__':
    opt = TestOptions().parse(verbose=False)
    os.makedirs('opts', exist_ok=True)
    if 'full' in opt.restore_G_path:
        output_path = 'opts/opt_full.pkl'
    else:
        assert 'compressed' in opt.restore_G_path
        output_path = 'opts/opt_compressed.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(opt, f)
    print('Save options at [%s].' % output_path)
