import torch
import torch.nn as nn
import logging

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

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

        for _, (data, targets) in enumerate(train_loader):
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
                for _, (data, targets) in enumerate(test_loader):
                    outputs = clf(data)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
            test_loss /= len(test_loader)
            logger.info("epoch : {}, train loss : {}, test loss : {}".format(
                epoch, train_loss, test_loss))
    logger.info('Classifier training complete.')

def evaluate_clf(clf, valid_loader, args):
    all_outputs = []
    all_targets = []

    clf.eval()
    with torch.no_grad():
        for _, (data, targets) in enumerate(valid_loader):
            outputs = clf(data)

            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    outputs_array = np.concatenate(all_outputs, axis=0)
    targets_array = np.concatenate(all_targets, axis=0)

    outputs_array = (outputs_array > .5).astype(int)

    metrics = []
    for i in range(targets_array.shape[1]):
        accuracy = accuracy_score(targets_array[:, i], outputs_array[:, i])
        precision = precision_score(targets_array[:, i], outputs_array[:, i])
        recall = recall_score(targets_array[:, i], outputs_array[:, i])
        f1 = f1_score(targets_array[:, i], outputs_array[:, i])
        try:
            auc = roc_auc_score(targets_array[:, i], outputs_array[:, i])
        except ValueError:
            auc = float('nan')
        metrics.append([accuracy, precision, recall, f1, auc])
    
    return np.array(metrics).mean(axis=0)
