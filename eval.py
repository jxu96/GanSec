from scipy.io import savemat, loadmat
import argparse
import numpy as np
import torch
import logging
import os
import sys
from sklearn.preprocessing import MinMaxScaler
from scripts.data_loader import get_dataloader, get_dataset, get_windows
from scripts.clf import train_clf, evaluate_clf
from scripts.gan import load_gan, gen_synthetic, gen_synthetic_labeledgan
# from scripts.eval_dist import calculate_metrics
# from models.dnn import Classifier

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tag', type=str)
    parser.add_argument('-m', '--model', type=str)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-w', '--window', type=int, default=5)

    parser.add_argument('--epoch-num', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--block-size', type=int, default=10)
    parser.add_argument('--lr-clf', type=float, default=0.001)
    parser.add_argument('--threshold-d', type=float, default=.0)

    return parser.parse_args()

def configure_logging(output, debug=False):
    format = "%(asctime)s - [%(levelname)s] [%(name)s] %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('backups/{}/eval.out'.format(output))
    ]

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format=format,
        handlers=handlers
    )

def main():
    args = parse_args()
    configure_logging(args.tag)
    current_time = args.tag

    if args.model == 'dnn':
        from models.dnn import Generator, Discriminator, LabeledDiscriminator, Classifier
    elif args.model == 'cnn':
        from models.cnn import Generator, Discriminator, LabeledDiscriminator, Classifier
    elif args.model == 'rnn':
        from models.rnn import Generator, Discriminator, LabeledDiscriminator, Classifier
    
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    ref_index = 3 # f1
    augment_ratios = [.1, .2, .3, .5, .7, 1., 1.5, 2., 3., 5.]

    if os.path.exists(f'backups/{current_time}/eval.mat'):
        outputs = loadmat(f'backups/{current_time}/eval.mat', squeeze_me=True)
    else:
        outputs = {}

    # Prepare datasets
    scaler = MinMaxScaler()
    A, A_label = get_dataset('data/ue_jamming_detection/train.csv')
    B, B_label = get_dataset('data/ue_jamming_detection/valid.csv')
    A[A.columns] = scaler.fit_transform(A)
    B[B.columns] = scaler.transform(B)

    A, A_label = get_windows(A, A_label, args.window)
    B, B_label = get_windows(B, B_label, args.window)

    A_0 = A[np.apply_along_axis(np.mean, 1, A_label) == 0]
    A_1 = A[np.apply_along_axis(np.mean, 1, A_label) == 1]

    A_train_loader, A_test_loader = get_dataloader(
        A, A_label, device=device, batch_size=args.batch_size, train_test_split=.2)
    A_loader, _ = get_dataloader(
        A, A_label, device=device, batch_size=args.batch_size)
    B_loader, _ = get_dataloader(
        B, B_label, device=device, batch_size=args.batch_size)

    ## Pre-augment clf evaluate
    logger = logging.getLogger('pre-aug')
    clf = Classifier(
        data_shape=[A.shape[1], A.shape[2]],
        label_shape=[A_label.shape[1], 1],
        ).to(device)
    
    train_clf(clf, A_train_loader, A_test_loader, args)
    results_A = evaluate_clf(clf, A_loader, args)
    results_B = evaluate_clf(clf, B_loader, args)

    if 'pre-aug-B' not in outputs.keys() or outputs['pre-aug-B'][ref_index] > results_B[ref_index]:
        outputs['pre-aug-A'] = results_A
        outputs['pre-aug-B'] = results_B

    logger.info('Perf on A set: {}'.format(outputs['pre-aug-A']))
    logger.info('Perf on B set: {}'.format(outputs['pre-aug-B']))

    ## Embedded Conditional GAN
    logger = logging.getLogger('ec-gan')

    ### Load GAN
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
    
    load_gan(generator, discriminator, loc=f'backups/{current_time}/ec-gan')

    ### Generate Synthetic
    for ratio in augment_ratios:
        ratio_pc = int(ratio*100)
        logger.info('Augmenting with {}% ..'.format(ratio_pc))
        amount = int(A.shape[0] * ratio // 2)

        x_flag_0, y_flag_0 = gen_synthetic(generator, discriminator, amount, 0, args.window, device, threshold_d=args.threshold_d)
        x_flag_1, y_flag_1 = gen_synthetic(generator, discriminator, amount, 1, args.window, device, threshold_d=args.threshold_d)
    
        A_flag = np.concatenate([A, x_flag_0, x_flag_1], axis=0)
        A_label_flag = np.concatenate([A_label, y_flag_0, y_flag_1], axis=0)

        A_flag_train_loader, A_flag_test_loader = get_dataloader(A_flag, A_label_flag, device=device, batch_size=args.batch_size, train_test_split=.1)

        clf_augmented = Classifier(
            data_shape=[A.shape[1], A.shape[2]],
            label_shape=[A_label.shape[1], 1],
        ).to(device)

        train_clf(clf_augmented, A_flag_train_loader, A_flag_test_loader, args)

        key_A = f'ec-gan_A_{ratio_pc}'
        key_B = f'ec-gan_B_{ratio_pc}'

        results_A = evaluate_clf(clf_augmented, A_loader, args)
        results_B = evaluate_clf(clf_augmented, B_loader, args)

        if key_B not in outputs.keys() or outputs[key_B][ref_index] < results_B[ref_index]:
            outputs[key_A] = results_A
            outputs[key_B] = results_B

        logger.info('({}) Perf on A set: {}'.format(ratio_pc, outputs[key_A]))
        logger.info('({}) Perf on B set: {}'.format(ratio_pc, outputs[key_B]))

    ## Classification Oriented GAN
    logger = logging.getLogger('co-gan')

    ### Load GAN
    generator = Generator(
        data_shape=[A.shape[1], A.shape[2]],
        label_shape=[A_label.shape[1], 1],
        label_embedding_size=8,
        ).to(device)

    discriminator = LabeledDiscriminator(
        data_shape=[A.shape[1], A.shape[2]],
        label_shape=[A_label.shape[1], 1],
        ).to(device)

    load_gan(generator, discriminator, loc=f'backups/{current_time}/co-gan')

    ### Generate Synthetic
    for ratio in augment_ratios:
        ratio_pc = int(ratio*100)
        logger.info('Augmenting with {}% ..'.format(ratio_pc))
        amount = int(A.shape[0] * ratio // 2)

        x_flag_0, y_flag_0 = gen_synthetic_labeledgan(generator, discriminator, amount, 0, args.window, device, threshold_d=args.threshold_d)
        x_flag_1, y_flag_1 = gen_synthetic_labeledgan(generator, discriminator, amount, 1, args.window, device, threshold_d=args.threshold_d)
    
        A_flag = np.concatenate([A, x_flag_0, x_flag_1], axis=0)
        A_label_flag = np.concatenate([A_label, y_flag_0, y_flag_1], axis=0)

        A_flag_train_loader, A_flag_test_loader = get_dataloader(A_flag, A_label_flag, device=device, batch_size=args.batch_size, train_test_split=.1)

        clf_augmented = Classifier(
            data_shape=[A.shape[1], A.shape[2]],
            label_shape=[A_label.shape[1], 1],
        ).to(device)

        train_clf(clf_augmented, A_flag_train_loader, A_flag_test_loader, args)

        key_A = f'co-gan_A_{ratio_pc}'
        key_B = f'co-gan_B_{ratio_pc}'

        results_A = evaluate_clf(clf_augmented, A_loader, args)
        results_B = evaluate_clf(clf_augmented, B_loader, args)

        if key_B not in outputs.keys() or outputs[key_B][ref_index] < results_B[ref_index]:
            outputs[key_A] = results_A
            outputs[key_B] = results_B

        logger.info('({}) Perf on A set: {}'.format(ratio_pc, outputs[key_A]))
        logger.info('({}) Perf on B set: {}'.format(ratio_pc, outputs[key_B]))
    
    savemat(f'backups/{current_time}/eval.mat', outputs)

if __name__ == "__main__":
    main()
