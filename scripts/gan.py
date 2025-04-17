import torch
import torch.nn as nn
import logging
import numpy as np

def train_gan(generator, discriminator, train_loader, test_loader, device, args):
    logger = logging.getLogger('train_gan')

    criterion = nn.BCELoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d)

    logger.info('Start training GAN ..')
    logger.info('[epoch_num_gan]: {}'.format(args.epoch_num_gan))
    logger.info('[lr-g]: {}, [lr-d]: {}'.format(args.lr_g, args.lr_d))

    for epoch in range(args.epoch_num_gan):
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
        
        train_loss_g /= len(train_loader)
        train_loss_d /= len(train_loader)

        if (epoch+1) % args.block_size_gan == 0 or epoch == 0:
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
            
            logger.info("epoch : {}, train loss d : {}, tranin loss g : {}, test loss d : {}, test loss g : {}".format(
                epoch, train_loss_d, train_loss_g, test_loss_d, test_loss_g))
    logger.info('GAN training complete.')

def train_labeledgan(generator, discriminator, train_loader, test_loader, device, args):
    logger = logging.getLogger('train_gan')

    criterion = nn.BCELoss()
    # criterion1 = nn.CrossEntropyLoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d)

    logger.info('Start training LabeledGAN ..')
    logger.info('[epoch_num_gan]: {}'.format(args.epoch_num_gan))
    logger.info('[lr-g]: {}, [lr-d]: {}'.format(args.lr_g, args.lr_d))

    for epoch in range(args.epoch_num_gan):
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
                criterion(pred_fake_class, label)
            loss_label = 0.0
            loss_d = (loss_real + loss_fake)/2 + loss_label
            train_loss_d += loss_d.item()
            optimizer_d.zero_grad()
            optimizer_g.zero_grad()
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

        if (epoch+1) % args.block_size_gan == 0 or epoch == 0:
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

            logger.info("epoch : {}, train loss d : {}, tranin loss g : {}, test loss d : {}, test loss g : {}".format(
                epoch, train_loss_d, train_loss_g, test_loss_d, test_loss_g))
    logger.info('GAN training complete.')

def gen_synthetic(generator, discriminator, amount, label, n_label, device, threshold_d=.3):
    logger = logging.getLogger('generate_synthetic')
    logger.info(f'Generating {amount} samples ({label}) ...')

    remaining = amount
    loop_count, safe_break = 0, 100
    synthetic_data = []
    while remaining > 0 or loop_count < safe_break:
        y = torch.full((remaining, n_label), label, device=device)
        out = generator.generate_random(remaining, device, y)
        likelihood = discriminator(out, y)
        is_realistic = torch.flatten(likelihood >= threshold_d)
        synthetic_data.append(out[is_realistic].detach().cpu().numpy())
        
        remaining -= sum(is_realistic)
        loop_count += 1
        logger.debug(f'remaining {remaining} ..')
    
    if remaining > 0:
        logger.warning(f'failed to generate enough realistic samples: [{amount-remaining}/{amount}]')

    return np.concatenate(synthetic_data, axis=0), np.full((amount-remaining, n_label), label)

def gen_synthetic_labeledgan(generator, discriminator, amount, label, n_label, device, threshold_d=.3):
    logger = logging.getLogger('generate_synthetic')
    logger.info(f'Generating {amount} samples ({label}) ...')

    remaining = amount
    loop_count, safe_break = 0, 100
    synthetic_data = []
    while remaining > 0 or loop_count < safe_break:
        y = torch.full((remaining, n_label), label, device=device)
        out = generator.generate_random(remaining, device, y)
        likelihood, _ = discriminator(out)
        is_realistic = torch.flatten(likelihood >= threshold_d)
        synthetic_data.append(out[is_realistic].detach().cpu().numpy())
        
        remaining -= sum(is_realistic)
        loop_count += 1
        logger.debug(f'remaining {remaining} ..')
    
    if remaining > 0:
        logger.warning(f'failed to generate enough realistic samples: [{amount-remaining}/{amount}]')

    return np.concatenate(synthetic_data, axis=0), np.full((amount-remaining, n_label), label)
