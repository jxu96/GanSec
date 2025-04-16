#!/usr/bin/env python3
import sys
import os
import time
import argparse
import logging
import torch
import pandas as pd
import numpy as np
# from models.cnn import Generator, Discriminator, LabeledDiscriminator, Classifier
from models.dnn import Generator, Discriminator, LabeledDiscriminator, Classifier
# from models.rnn import Generator, Discriminator, LabeledDiscriminator, Classifier

from scripts.clf import train_clf, evaluate_clf
from scripts.gan import train_gan, train_labeledgan, gen_synthetic
from scripts.data_loader import get_dataloader, get_windows
from scripts.eval_dist import calculate_metrics
from sklearn.preprocessing import MinMaxScaler

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
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

def configure_logging(debug=False):
    format = "%(asctime)s - [%(levelname)s] [%(name)s] %(message)s"
    current_time = time.asctime()
    logging_file = '{}/logs/{}.out'.format(
        os.path.dirname(__file__), current_time.replace(' ', '_'))
    os.makedirs(os.path.dirname(logging_file), exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(logging_file)
    ]

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format=format,
        handlers=handlers
    )

def get_dataset(file_path):
    df = pd.read_csv(file_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    label = df['Label'].values
    df.drop(columns='Label', inplace=True)

    return df, label

def main():
    args = parse_args()
    configure_logging(debug=args.verbose)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    scaler = MinMaxScaler()
    A, A_label = get_dataset('data/ue_jamming_detection/train.csv')
    B, B_label = get_dataset('data/ue_jamming_detection/valid.csv')
    A[A.columns] = scaler.fit_transform(A)
    B[B.columns] = scaler.transform(B)

    A, A_label = get_windows(A, A_label, args.window)
    B, B_label = get_windows(B, B_label, args.window)

    A_train_loader, A_test_loader = get_dataloader(
        A, A_label, device=device, batch_size=args.batch_size, train_test_split=.2)
    A_loader, _ = get_dataloader(
        A, A_label, device=device, batch_size=args.batch_size)
    B_loader, _ = get_dataloader(
        B, B_label, device=device, batch_size=args.batch_size)

    ## Pre-augment clf evaluate
    logger = logging.getLogger('pre-eval')
    clf = Classifier(
        data_shape=[A.shape[1], A.shape[2]],
        label_shape=[A_label.shape[1], 1],
        ).to(device)
    
    train_clf(clf, A_train_loader, A_test_loader, args)
    results = evaluate_clf(clf, B_loader, args)
    logger.info(results)

    ## Train GAN
    generator = Generator(
        data_shape=[A.shape[1], A.shape[2]],
        label_shape=[A_label.shape[1], 1],
        ).to(device)

    discriminator = Discriminator(
        data_shape=[A.shape[1], A.shape[2]],
        label_shape=[A_label.shape[1], 1],
        ).to(device)
    
    train_gan(generator, discriminator, A_train_loader, A_test_loader, device, args)
    
    labeled_discriminator = LabeledDiscriminator(
        data_shape=[A.shape[1], A.shape[2]],
        label_shape=[A_label.shape[1], 1],
        ).to(device)

    train_labeledgan(generator, labeled_discriminator, A_train_loader, A_test_loader, device, args)

    ## Generate Synthetic (Mode 1)
    logger = logging.getLogger('gan-1')
    x_flag_0, y_flag_0 = gen_synthetic(generator, discriminator, 10000, 0, args.window, device)
    x_flag_1, y_flag_1 = gen_synthetic(generator, discriminator, 10000, 1, args.window, device)
    x_flag = np.concatenate([x_flag_0, x_flag_1], axis=0)
    y_flag = np.concatenate([y_flag_0, y_flag_1], axis=0)
    
    calculate_metrics(A, x_flag)
    synthetic_train_loader, synthetic_test_loader = get_dataloader(x_flag, y_flag, device=device, batch_size=args.batch_size, train_test_split=.1)

    ## Post-augment clf evaluate
    clf_augmented = Classifier(
        data_shape=[A.shape[1], A.shape[2]],
        label_shape=[A_label.shape[1], 1],
    ).to(device)

    train_clf(clf_augmented, synthetic_train_loader, synthetic_test_loader, args)
    results = evaluate_clf(clf, A_loader, args)
    logger.info(f'Perf on A set: {results}')
    results = evaluate_clf(clf, B_loader, args)
    logger.info(f'Perf on B set: {results}')

    ## Generate Synthetic (Mode 2)
    logger = logging.getLogger('gan-2')
    x_flag_0, y_flag_0 = gen_synthetic(generator, discriminator, 10000, 0, args.window, device)
    x_flag_1, y_flag_1 = gen_synthetic(generator, discriminator, 10000, 1, args.window, device)
    x_flag = np.concatenate([x_flag_0, x_flag_1], axis=0)
    y_flag = np.concatenate([y_flag_0, y_flag_1], axis=0)
    
    calculate_metrics(A, x_flag)
    synthetic_train_loader, synthetic_test_loader = get_dataloader(x_flag, y_flag, device=device, batch_size=args.batch_size, train_test_split=.1)

    ## Post-augment clf evaluate
    clf_augmented = Classifier(
        data_shape=[A.shape[1], A.shape[2]],
        label_shape=[A_label.shape[1], 1],
    ).to(device)

    train_clf(clf_augmented, synthetic_train_loader, synthetic_test_loader, args)
    results = evaluate_clf(clf, A_loader, args)
    logger.info(f'Perf on A set: {results}')
    results = evaluate_clf(clf, B_loader, args)
    logger.info(f'Perf on B set: {results}')

    # train_set, train_label_set = get_windows(train, train_label, window_size=args.window)
    # synthetic_set = generator.generate_random(train_set.shape[0], device, torch.Tensor(train_label_set)).detach().numpy()
    # metrics = calculate_metrics(train_set, synthetic_set)

if __name__ == "__main__":
    main()
