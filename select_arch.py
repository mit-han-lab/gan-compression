import argparse
import pickle


def takeMACs(item):
    return item['macs']


def main(opt):
    with open(opt.pkl_path, 'rb') as f:
        results = pickle.load(f)
    results.sort(key=takeMACs)

    for item in results:
        assert isinstance(item, dict)
        qualified = True
        if item['macs'] > opt.macs:
            qualified = False
        elif 'fid' in item and item['fid'] > opt.fid:
            qualified = False
        elif 'mAP' in item and item['mAP'] < opt.mAP:
            qualified = False
        if qualified:
            print(item)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='An auxiliary script to read the parse the output pickle and select the configurations you want.')
    parser.add_argument('--pkl_path', type=str, required=True, help='the input .pkl file path')
    parser.add_argument('--macs', type=float, default=5.68e9, help='the MACs threshold')
    parser.add_argument('--fid', type=float, default=-1, help='the FID threshold')
    parser.add_argument('--mAP', type=float, default=1e18, help='the mAP threshold')
    opt = parser.parse_args()
    main(opt)
