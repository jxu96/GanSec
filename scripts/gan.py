import torch
import torch.nn as nn
import logging

def train_clf(clf, train_loader, test_loader, args):
    logger = logging.getLogger('train_clf')

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(clf.parameters(), lr=args.lr_clf)

    logger.info('Start training Classifier ..')
    logger.info('[epoch_num]: {}'.format(args.epoch_num))
    logger.info('[lr_clf]: {}'.format(args.lr_clf))

    for epoch in range(args.epoch_num):
        train_loss = .0
        clf.train()

        for batch_idx, (data, targets) in enumerate(train_loader):
            outputs = clf(data)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss /= len(train_loader)

        if (epoch+1) % args.block_size == 0 or epoch == 0:
            test_loss = .0
            clf.eval()
            
            with torch.no_grad():
                for data, targets in test_loader:
                    outputs = clf(data)
                    loss = criterion(outputs, targets)
                    
                    test_loss += loss.item()
            test_loss /= len(test_loader)
            logger.info("epoch : {}, train loss : {}, test loss : {}".format(
                epoch, train_loss, test_loss))
    logger.info('Classifier training complete.')

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

            logger.info("epoch : {}, train loss d : {}, tranin loss g : {}, test loss d : {}, test loss g : {}".format(
                epoch, train_loss_d, train_loss_g, test_loss_d, test_loss_g))
    logger.info('GAN training complete.')