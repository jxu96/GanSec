#!/usr/bin/env python3
import sys
import os
import time
import argparse
import logging
import torch
import torch.nn as nn
from models.cnn_gan import Generator, Discriminator
# from models.dnn_gan import Generator, Discriminator
# from models.rnn_gan import Generator, Discriminator
from scripts.data_loader import get_dataloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-w', '--window', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument('--block_size', type=int, default=10)

    return parser.parse_args()


def configure_logging(debug=False):
    format = "%(asctime)s - [%(levelname)s] [%(name)s] %(message)s"
    current_time = time.asctime()
    logging_file = '{}/logs/{}.out'.format(
        os.path.dirname(__file__), current_time.replace(' ', '_'))
    handlers = [
        logging.StreamHandler(sys.stdout),
        # logging.FileHandler(logging_file)
    ]
    os.makedirs(os.path.dirname(logging_file), exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format=format,
        handlers=handlers
    )


def main():
    args = parse_args()
    configure_logging(debug=args.verbose)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    train_loader, test_loader = get_dataloader(
        'data/ue_jamming_detection/train.csv', window_size=args.window, device=device, batch_size=args.batch_size, train_test_split=.2)
    # valid, _ = get_dataloader('data/ue_jamming_detection/valid.csv', window_size=args.window, device=device, batch_size=args.batch_size)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCELoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

    for epoch in range(args.epoch_num):
        train_loss_g = 0.0
        train_loss_d = 0.0
        generator.train()
        discriminator.train()
        for _, (input, label) in enumerate(train_loader):
            input, label = input.to(device), label.to(device)
            cur_size = input.shape[0]
            y_real = torch.ones((label.shape[0], 1)).to(device)
            y_fake = torch.zeros((label.shape[0], 1)).to(device)

            #
            pred_real = discriminator(input, label)
            loss_real = criterion(pred_real, y_real)
            X_fake = generator.generate_random(cur_size, device, label)
            pred_fake = discriminator(X_fake, label)
            loss_fake = criterion(pred_fake, y_fake)
            loss_label = 0.0
            loss_d = (loss_real + loss_fake)/2 + loss_label
            train_loss_d += loss_d.item()
            optimizer_d.zero_grad()
            optimizer_g.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            #
            fake_sample = generator.generate_random(cur_size, device, label)
            pred_fake = discriminator(fake_sample, label)
            loss_g = criterion(pred_fake, y_real)
            train_loss_g += loss_g.item()
            optimizer_d.zero_grad()
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

        if (epoch+1) % args.block_size == 0 or epoch == 0:
            test_loss_g = 0.0
            test_loss_d = 0.0
            generator.eval()
            discriminator.eval()
            with torch.no_grad():
                for _, (input_test, label_test) in enumerate(test_loader):
                    input_test, label_test = input_test.to(
                        device), label_test.to(device)
                    cur_size_test = input_test.shape[0]
                    # print("test", cur_size_test,
                    #       input_test.shape, label_test.shape)
                    y_real_test = torch.ones(
                        (label_test.shape[0], 1)).to(device)
                    y_fake_test = torch.zeros(
                        (label_test.shape[0], 1)).to(device)

                    #
                    preal = discriminator(input_test, label_test)
                    lreal = criterion(preal, y_real_test)
                    xfake = generator.generate_random(
                        cur_size_test, device, label_test)
                    pfake = discriminator(xfake, label_test)
                    lfake = criterion(pfake, y_fake_test)
                    ldtest = (lreal+lfake)/2.
                    test_loss_d += ldtest.item()

                    #
                    xfake = generator.generate_random(
                        cur_size_test, device, label_test)
                    pfake = discriminator(xfake, label_test)
                    lgtest = criterion(pfake, y_real_test)
                    test_loss_g += lgtest.item()

            print("epoch : {}, train loss d : {}, tranin loss g : {}, test loss d : {}, test loss g : {}".format(
                epoch, train_loss_d, train_loss_g, test_loss_d, test_loss_g))


if __name__ == "__main__":
    main()
