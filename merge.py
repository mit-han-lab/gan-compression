import argparse
import os
import pickle


def main(input_dir, output_path):
    files = os.listdir(input_dir)
    results = []
    for file in files:
        if file.endswith('.pkl'):
            with open(os.path.join(input_dir, file), 'rb') as f:
                result = pickle.load(f)
                results += result
    with open(opt.output_path, 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    opt = parser.parse_args()
    main(opt.input_dir, opt.output_path)
