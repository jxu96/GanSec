from scipy.io import savemat, loadmat
import argparse
import numpy as np
import torch
import os
import sys
from sklearn.preprocessing import MinMaxScaler
from scripts.data_loader import get_dataset, get_windows
from scripts.gan import load_gan, gen_synthetic, gen_synthetic_labeledgan
from scripts.eval_dist import calculate_fid_tabular, calculate_mmd_rbf
# from models.dnn import Classifier

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tag', type=str)
    parser.add_argument('-m', '--model', type=str)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-w', '--window', type=int, default=5)

    return parser.parse_args()

def main():
    args = parse_args()
    current_time = args.tag

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

    if os.path.exists(f'backups/{current_time}/dist.mat'):
        outputs = loadmat(f'backups/{current_time}/dist.mat', squeeze_me=True)
    else:
        outputs = {}

    eval_loop = 100

    # Prepare datasets
    scaler = MinMaxScaler()
    A, A_label = get_dataset('data/ue_jamming_detection/train.csv')
    A[A.columns] = scaler.fit_transform(A)

    A, A_label = get_windows(A, A_label, args.window)

    A_0 = A[np.apply_along_axis(np.mean, 1, A_label) == 0]
    A_1 = A[np.apply_along_axis(np.mean, 1, A_label) == 1]

    # A_train_loader, A_test_loader = get_dataloader(
    #     A, A_label, device=device, batch_size=args.batch_size, train_test_split=.2)
    # A_loader, _ = get_dataloader(
    #     A, A_label, device=device, batch_size=args.batch_size)
    # B_loader, _ = get_dataloader(
    #     B, B_label, device=device, batch_size=args.batch_size)

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
    
    load_gan(ec_gen, ec_dis, loc=f'backups/{current_time}/ec-gan')

    co_gen = Generator(
        data_shape=[A.shape[1], A.shape[2]],
        label_shape=[A_label.shape[1], 1],
        label_embedding_size=8,
        ).to(device)

    co_dis = LabeledDiscriminator(
        data_shape=[A.shape[1], A.shape[2]],
        label_shape=[A_label.shape[1], 1],
        ).to(device)

    load_gan(co_gen, co_dis, loc=f'backups/{current_time}/co-gan')

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

    for i in range(eval_loop):
        print(f'{i}/{eval_loop}')
        x_flag_0, _ = gen_synthetic(ec_gen, ec_dis, A_0.shape[0], 0, args.window, device)
        results['ec-gan_0_fid'].append(calculate_fid_tabular(A_0, x_flag_0))
        results['ec-gan_0_mmd'].append(calculate_mmd_rbf(A_0, x_flag_0))

        x_flag_1, _ = gen_synthetic(ec_gen, ec_dis, A_1.shape[0], 1, args.window, device)
        results['ec-gan_1_fid'].append(calculate_fid_tabular(A_1, x_flag_1))
        results['ec-gan_1_mmd'].append(calculate_mmd_rbf(A_1, x_flag_1))

        x_flag_0, _ = gen_synthetic_labeledgan(co_gen, co_dis, A_0.shape[0], 0, args.window, device)
        results['co-gan_0_fid'].append(calculate_fid_tabular(A_0, x_flag_0))
        results['co-gan_0_mmd'].append(calculate_mmd_rbf(A_0, x_flag_0))

        x_flag_1, _ = gen_synthetic_labeledgan(co_gen, co_dis, A_1.shape[0], 1, args.window, device)
        results['co-gan_1_fid'].append(calculate_fid_tabular(A_1, x_flag_1))
        results['co-gan_1_mmd'].append(calculate_mmd_rbf(A_1, x_flag_1))
    
    for key, result in results.items():
        print('[{}]: Mean {}'.format(key, np.mean(result)))
        if key in outputs:
            outputs[key] += result

            outputs[key].sort()
            outputs[key] = outputs[key][:eval_loop]

        else:
            outputs[key] = result

    savemat(f'backups/{current_time}/dist.mat', outputs)

if __name__ == "__main__":
    main()
