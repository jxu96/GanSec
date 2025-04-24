from scipy.io import savemat, loadmat
import argparse
import numpy as np
import torch
import os
import logging
import sys
from sklearn.preprocessing import MinMaxScaler
from scripts.data_loader import get_dataset, get_windows
from scripts.gan import load_gan, ec_gan_gen, co_gan_gen
from scripts.eval_dist import calculate_fid_tabular, calculate_mmd_rbf

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tag', type=str)
    parser.add_argument('-m', '--model', type=str)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-w', '--window', type=int, default=5)
    parser.add_argument('-l', '--loop', type=int, default=100)

    return parser.parse_args()

def configure_logging(output, debug=False):
    format = "%(asctime)s - [%(levelname)s] [%(name)s] %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output)
    ]

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format=format,
        handlers=handlers
    )

def main():
    args = parse_args()
    current_time = args.tag
    folder = f'backups/{current_time}_{args.model.upper()}'

    configure_logging(output=f'{folder}/dist.out', debug=args.verbose)
    logging.info(args)

    if args.model == 'dnn':
        from models.dnn import Generator, Discriminator, LabeledDiscriminator
    elif args.model == 'cnn':
        from models.cnn import Generator, Discriminator, LabeledDiscriminator
    elif args.model == 'rnn':
        from models.rnn import Generator, Discriminator, LabeledDiscriminator
    
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    if os.path.exists(f'{folder}/dist.mat'):
        outputs = loadmat(f'{folder}/dist.mat', squeeze_me=True)
    else:
        outputs = {}

    # Prepare datasets
    scaler = MinMaxScaler()
    A, A_label = get_dataset('data/ue_jamming_detection/train.csv')
    A[A.columns] = scaler.fit_transform(A)

    A, A_label = get_windows(A, A_label, args.window)

    A_0 = A[np.apply_along_axis(np.mean, 1, A_label) == 0]
    A_1 = A[np.apply_along_axis(np.mean, 1, A_label) == 1]

    # Load GAN
    ec_gen = Generator(
        data_shape=[A.shape[1], A.shape[2]],
        label_shape=[A_label.shape[1], 1],
        label_embedding_size=8,
        ).to(device)
    
    ec_dis = Discriminator(
        data_shape=[A.shape[1], A.shape[2]],
        label_shape=[A_label.shape[1], 1],
        label_embedding_size=8,
        ).to(device)
    
    load_gan(ec_gen, ec_dis, loc=f'{folder}/ec-gan')

    co_gen = Generator(
        data_shape=[A.shape[1], A.shape[2]],
        label_shape=[A_label.shape[1], 1],
        label_embedding_size=8,
        ).to(device)

    co_dis = LabeledDiscriminator(
        data_shape=[A.shape[1], A.shape[2]],
        label_shape=[A_label.shape[1], 1],
        ).to(device)

    load_gan(co_gen, co_dis, loc=f'{folder}/co-gan')

    results = {
        'ec-gan_0_fid': [],
        'ec-gan_1_fid': [],
        'co-gan_0_fid': [],
        'co-gan_1_fid': [],
        'ec-gan_0_mmd': [],
        'ec-gan_1_mmd': [],
        'co-gan_0_mmd': [],
        'co-gan_1_mmd': [],
    }

    block = args.loop // 10

    for i in range(args.loop):
        x_flag_0, _ = ec_gan_gen(ec_gen, ec_dis, A_0.shape[0], 0, args.window, device)
        results['ec-gan_0_fid'].append(calculate_fid_tabular(A_0, x_flag_0))
        results['ec-gan_0_mmd'].append(calculate_mmd_rbf(A_0, x_flag_0))

        x_flag_1, _ = ec_gan_gen(ec_gen, ec_dis, A_1.shape[0], 1, args.window, device)
        results['ec-gan_1_fid'].append(calculate_fid_tabular(A_1, x_flag_1))
        results['ec-gan_1_mmd'].append(calculate_mmd_rbf(A_1, x_flag_1))

        x_flag_0, _ = co_gan_gen(co_gen, co_dis, A_0.shape[0], 0, args.window, device)
        results['co-gan_0_fid'].append(calculate_fid_tabular(A_0, x_flag_0))
        results['co-gan_0_mmd'].append(calculate_mmd_rbf(A_0, x_flag_0))

        x_flag_1, _ = co_gan_gen(co_gen, co_dis, A_1.shape[0], 1, args.window, device)
        results['co-gan_1_fid'].append(calculate_fid_tabular(A_1, x_flag_1))
        results['co-gan_1_mmd'].append(calculate_mmd_rbf(A_1, x_flag_1))

        if i == 0 or (i+1) % block == 0:
            logging.info('Loop {}/{}:\n[EC-GAN] FID: {} MMD: {}\n[CO-GAN] FID: {} MMD: {}'.format(
                i+1, args.loop,
                np.mean(results['ec-gan_0_fid'] + results['ec-gan_1_fid']),
                np.mean(results['ec-gan_0_mmd'] + results['ec-gan_1_mmd']),
                np.mean(results['co-gan_0_fid'] + results['co-gan_1_fid']),
                np.mean(results['co-gan_0_mmd'] + results['co-gan_1_mmd'])))
    
    for key, result in results.items():
        logging.info('[{}]: Mean {}'.format(key, np.mean(result)))
        if key in outputs:
            outputs[key] += result
            # outputs[key].sort()
            # outputs[key] = outputs[key][:args.loop]

        else:
            outputs[key] = result

    savemat(f'{folder}/dist.mat', outputs)

if __name__ == "__main__":
    main()
