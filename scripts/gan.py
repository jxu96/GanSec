import torch
import torch.nn as nn
import logging
import numpy as np
import os

def save_gan(generator, discriminator, loc):
    os.makedirs(loc, exist_ok=True)
    torch.save(generator.state_dict(), f'{loc}/gen.pth')
    torch.save(discriminator.state_dict(), f'{loc}/dis.pth')

def load_gan(generator, discriminator, loc):
    generator.load_state_dict(torch.load(f'{loc}/gen.pth', weights_only=True, map_location=torch.device('cpu')))
    discriminator.load_state_dict(torch.load(f'{loc}/dis.pth', weights_only=True, map_location=torch.device('cpu')))

def train_ec_gan(generator, discriminator, train_loader, test_loader, device, args):
    logger = logging.getLogger('train_gan')
    traces = {
        'epochs': [],
        'loss_d': [],
        'loss_g': [],
        'sample_g': [],
    }

    if args.ec_epoch_num <= 0:
        return

    criterion = nn.BCELoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.ec_lr_g)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.ec_lr_d)

    logger.info('Start training EC-GAN ..')
    logger.info('[epoch_num]: {}'.format(args.ec_epoch_num))
    logger.info('[lr-g]: {}, [lr-d]: {}'.format(args.ec_lr_g, args.ec_lr_d))

    block_size = args.ec_block_size if args.ec_block_size > 0 else (args.ec_epoch_num // 10)
    # prune_d = args.epoch_num_gan // 2
    prune_d = -1

    for epoch in range(args.ec_epoch_num):
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
            loss_d = .4 * loss_real + .3 * loss_fake

            train_loss_d += loss_d.item()
            optimizer_d.zero_grad()
            optimizer_g.zero_grad()
            if prune_d < 0 or epoch < prune_d:
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
        
        train_loss_g /= len(train_loader)
        train_loss_d /= len(train_loader)

        if (epoch+1) % block_size == 0 or epoch == 0:
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

            test_loss_g /= len(test_loader)
            test_loss_d /= len(test_loader)
            traces['epochs'].append(epoch+1)
            traces['loss_d'].append(test_loss_d)
            traces['loss_g'].append(test_loss_g)
            traces['sample_g'].append(xfake.detach().cpu().numpy())
            
            logger.info("epoch : {}, train loss d : {}, train loss g : {}, test loss d : {}, test loss g : {}".format(
                epoch, train_loss_d, train_loss_g, test_loss_d, test_loss_g))
    logger.info('GAN training complete.')
    return traces

def train_co_gan(generator, discriminator, train_loader, test_loader, device, args):
    logger = logging.getLogger('train_gan')
    traces = {
        'epochs': [],
        'loss_d': [],
        'loss_g': [],
        'sample_g': [],
    }

    if args.co_epoch_num <= 0:
        return

    criterion = nn.BCELoss()
    # criterion1 = nn.CrossEntropyLoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.co_lr_g)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.co_lr_d)

    logger.info('Start training CO-GAN ..')
    logger.info('[epoch_num]: {}'.format(args.co_epoch_num))
    logger.info('[lr-g]: {}, [lr-d]: {}'.format(args.co_lr_g, args.co_lr_d))

    block_size = args.co_block_size if args.co_block_size > 0 else (args.co_epoch_num // 10)
    # prune_d = args.epoch_num_gan // 2
    prune_d = -1

    for epoch in range(args.co_epoch_num):
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
            pred_real, pred_real_class = discriminator(input)
            # import pdb;pdb.set_trace()
            loss_real = criterion(pred_real, y_real) + \
                criterion(pred_real_class, label)
            X_fake = generator.generate_random(cur_size, device, label)
            pred_fake, pred_fake_class = discriminator(X_fake)
            loss_fake = criterion(pred_fake, y_fake) + \
                .0 * criterion(pred_fake_class, label)
            
            loss_d = .4 * loss_real + .3 * loss_fake

            train_loss_d += loss_d.item()
            optimizer_d.zero_grad()
            optimizer_g.zero_grad()
            if prune_d < 0 or epoch < prune_d:
                loss_d.backward()
                optimizer_d.step()

            #
            fake_sample = generator.generate_random(cur_size, device, label)
            pred_fake, pred_fake_class = discriminator(fake_sample)
            loss_g = criterion(pred_fake, y_real) + \
                criterion(pred_fake_class, label)
            train_loss_g += loss_g.item()
            optimizer_d.zero_grad()
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

        train_loss_g /= len(train_loader)
        train_loss_d /= len(train_loader)

        if (epoch+1) % block_size == 0 or epoch == 0:
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
                    preal, preal_class = discriminator(input_test)
                    lreal = criterion(preal, y_real_test) + \
                        criterion(preal_class, label_test)
                    xfake = generator.generate_random(
                        cur_size_test, device, label_test)
                    pfake, pfake_class = discriminator(xfake)
                    lfake = criterion(pfake, y_fake_test) + \
                        criterion(pfake_class, label_test)
                    ldtest = (lreal+lfake)/2.
                    test_loss_d += ldtest.item()

                    #
                    xfake = generator.generate_random(
                        cur_size_test, device, label_test)
                    pfake, pfake_class = discriminator(xfake)
                    lgtest = criterion(pfake, y_real_test) + \
                        criterion(pfake_class, label_test)
                    test_loss_g += lgtest.item()

            test_loss_g /= len(test_loader)
            test_loss_d /= len(test_loader)
            traces['epochs'].append(epoch+1)
            traces['loss_d'].append(test_loss_d)
            traces['loss_g'].append(test_loss_g)
            traces['sample_g'].append(xfake.detach().cpu().numpy())

            logger.info("epoch : {}, train loss d : {}, train loss g : {}, test loss d : {}, test loss g : {}".format(
                epoch, train_loss_d, train_loss_g, test_loss_d, test_loss_g))
    logger.info('GAN training complete.')
    return traces

def ec_gan_gen(generator, discriminator, amount, label, n_label, device, threshold_d=.0):
    logger = logging.getLogger('generate_synthetic')
    logger.info(f'Generating {amount} samples ({label}) ...')

    remaining = amount
    loop_count, safe_break = 0, 100
    synthetic_data = []
    all_likelihoods = []

    while remaining > 0 and loop_count < safe_break:
        y = torch.full((remaining, n_label), label, device=device)
        out = generator.generate_random(remaining, device, y)
        likelihood = discriminator(out, y)
        is_realistic = torch.flatten(likelihood >= threshold_d)
        synthetic_data.append(out[is_realistic].detach().cpu().numpy())
        
        all_likelihoods.append(likelihood.detach().cpu().numpy())
        remaining -= sum(is_realistic)
        loop_count += 1
        logger.debug(f'remaining {remaining} ..')
    
    if remaining > 0:
        logger.warning(f'failed to generate enough realistic samples: [{amount-remaining}/{amount}]')
    
    if all_likelihoods:
        avg_likelihood = np.concatenate(all_likelihoods).mean()
        logger.info(f'Average likelihood of generated samples: {avg_likelihood:.4f}')

    return np.concatenate(synthetic_data, axis=0), np.full((amount-remaining, n_label), label)

def co_gan_gen(generator, discriminator, amount, label, n_label, device, threshold_d=.0):
    logger = logging.getLogger('generate_synthetic')
    logger.info(f'Generating {amount} samples ({label}) ...')

    remaining = amount
    loop_count, safe_break = 0, 100
    synthetic_data = []
    all_likelihoods = []

    while remaining > 0 and loop_count < safe_break:
        y = torch.full((remaining, n_label), label, device=device)
        out = generator.generate_random(remaining, device, y)
        likelihood, _ = discriminator(out)
        is_realistic = torch.flatten(likelihood >= threshold_d)
        synthetic_data.append(out[is_realistic].detach().cpu().numpy())
        
        all_likelihoods.append(likelihood.detach().cpu().numpy())
        remaining -= sum(is_realistic)
        loop_count += 1
        logger.debug(f'remaining {remaining} ..')
    
    if remaining > 0:
        logger.warning(f'failed to generate enough realistic samples: [{amount-remaining}/{amount}]')

    if all_likelihoods:
        avg_likelihood = np.concatenate(all_likelihoods).mean()
        logger.info(f'Average likelihood of generated samples: {avg_likelihood:.4f}')

    return np.concatenate(synthetic_data, axis=0), np.full((amount-remaining, n_label), label)
