#!/usr/bin/env python3
import sys
import os
import time
import argparse
import logging
import torch
from scripts.gan import train_gan, train_labeledgan, save_gan
from scripts.data_loader import get_dataloader, get_windows, get_dataset
from sklearn.preprocessing import MinMaxScaler

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-m', '--model', type=str)
    parser.add_argument('-w', '--window', type=int, default=5)

    parser.add_argument('--epoch-num', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--block-size', type=int, default=10)
    parser.add_argument('--lr-clf', type=float, default=0.001)

    parser.add_argument('--epoch-num-gan', type=int, default=100)
    parser.add_argument('--batch-size-gan', type=int, default=64)
    parser.add_argument('--block-size-gan', type=int, default=10)
    parser.add_argument('--lr-g', type=float, default=0.002)
    parser.add_argument('--lr-d', type=float, default=0.002)

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
    current_time = time.asctime().replace(' ', '_')
    os.makedirs(f'backups/{current_time}', exist_ok=True)

    configure_logging(output=f'backups/{current_time}/logs.out', debug=args.verbose)
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
    logging.info(f'using device: {device}')

    scaler = MinMaxScaler()
    A, A_label = get_dataset('data/ue_jamming_detection/train.csv')
    B, B_label = get_dataset('data/ue_jamming_detection/valid.csv')
    A[A.columns] = scaler.fit_transform(A)
    B[B.columns] = scaler.transform(B)

    A, A_label = get_windows(A, A_label, args.window)
    B, B_label = get_windows(B, B_label, args.window)

    # A_0 = A[np.apply_along_axis(np.mean, 1, A_label) == 0]
    # A_1 = A[np.apply_along_axis(np.mean, 1, A_label) == 1]

    A_train_loader, A_test_loader = get_dataloader(
        A, A_label, device=device, batch_size=args.batch_size, train_test_split=.2)
    # A_loader, _ = get_dataloader(
    #     A, A_label, device=device, batch_size=args.batch_size)
    # B_loader, _ = get_dataloader(
    #     B, B_label, device=device, batch_size=args.batch_size)

    ### Train GAN
    generator = Generator(
        data_shape=[A.shape[1], A.shape[2]],
        label_shape=[A_label.shape[1], 1],
        label_embedding_size=8,
        ).to(device)

    discriminator = Discriminator(
        data_shape=[A.shape[1], A.shape[2]],
        label_shape=[A_label.shape[1], 1],
        label_embedding_size=8,
        ).to(device)
    
    train_gan(generator, discriminator, A_train_loader, A_test_loader, device, args)
    save_gan(generator, discriminator, loc=f'backups/{current_time}/ec-gan')

    ### Train Labeled GAN
    generator = Generator(
        data_shape=[A.shape[1], A.shape[2]],
        label_shape=[A_label.shape[1], 1],
        label_embedding_size=8,
        ).to(device)

    discriminator = LabeledDiscriminator(
        data_shape=[A.shape[1], A.shape[2]],
        label_shape=[A_label.shape[1], 1],
        ).to(device)

    train_labeledgan(generator, discriminator, A_train_loader, A_test_loader, device, args)
    save_gan(generator, discriminator, loc=f'backups/{current_time}/co-gan')

if __name__ == "__main__":
    main()
