from backbone import Generator, Discriminator
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import os

class GANModel():
    def __init__(self, **config):
        self.logger = logging.getLogger('GAN')
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.logger.debug(f'using device: "{self.device}"')

        self.gen = Generator(**self.params).to(self.device)
        self.dis = Discriminator(**self.params_d).to(self.device)
            
        self._init_loss_function()
        self._init_optimizer()

    def _init_loss_function(self):
        def match_loss_fn(s):
            match s:
                case 'bce':
                    return nn.BCELoss()
                case _:
                    return nn.L1Loss()

        self.criterion = match_loss_fn(self.params.get('loss'))
        self.logger.info(f'using criterion: {self.criterion.__class__.__name__}')

        if hasattr(self, 'params_d'):
            self.criterion_d = match_loss_fn(self.params_d.get('loss'))
            self.logger.info(f'using criterion: {self.criterion_d.__class__.__name__} [D]')
    
    def _init_optimizer(self):
        def match_optimizer(s, params, lr):
            match s:
                case 'adam':
                    return torch.optim.Adam(params, lr=lr)
                case _:
                    return torch.optim.SGD(params, lr=lr)

        _opt, _lr = self.params.get('optimizer'), self.params.get('learn_rate')
        self.optimizer = match_optimizer(_opt, self.model.parameters(), _lr)
        self.logger.info(f'using optimizer: {self.optimizer.__class__.__name__} (learn rate: {_lr})')

        if hasattr(self, 'params_d'):
            _opt, _lr = self.params_d.get('optimizer'), self.params_d.get('learn_rate')
            self.optimizer_d = match_optimizer(_opt, self.model_d.parameters(), _lr)
            self.logger.info(f'using optimizer: {self.optimizer_d.__class__.__name__} [D] (learn rate: {_lr})')

    def _save_model_proc(self, save_loc):
        _file, _ext = os.path.splitext(save_loc)

        torch.save(self.model.state_dict(), save_loc)
        torch.save(self.optimizer.state_dict(), _file+f'_{self.optimizer.__class__.__name__}'+_ext)

        if hasattr(self, 'params_d'):
            torch.save(self.model_d.state_dict(), _file+'_discriminator'+_ext)
            torch.save(self.optimizer_d.state_dict(), _file+f'_discriminator_{self.optimizer_d.__class__.__name__}'+_ext)

    def _load_model_proc(self, load_loc):
        _file, _ext = os.path.splitext(load_loc)

        self.model.load_state_dict(torch.load(load_loc, weights_only=True))
        self.optimizer.load_state_dict(torch.load(_file+f'_{self.optimizer.__class__.__name__}'+_ext, weights_only=True))

        if hasattr(self, 'params_d'):
            self.model_d.load_state_dict(torch.load(_file+'_discriminator'+_ext, weights_only=True))
            self.optimizer_d.load_state_dict(torch.load(_file+f'_discriminator_{self.optimizer_d.__class__.__name__}'+_ext, weights_only=True))
        
    def _input_to_dataloader(self, input, label=None, split=.0):
        steps = self.params.get('3d_steps')
        try:
            steps = int(steps)
        except:
            steps = 0

        if steps: # Turn to 3D
            input = np.array([input[i:(i + steps)] for i in range(len(input) - steps + 1)])
            label = label if label is None else label[(steps - 1):]
        
        if label is None:
            input_tensor = torch.tensor(input, dtype=torch.float32, device=self.device)
        else:
            input_tensor = TensorDataset(torch.tensor(input, dtype=torch.float32, device=self.device), 
                                         torch.tensor(label, dtype=torch.float32, device=self.device))

        batch_size = self.params.get('batch_size')
        self.logger.info(f'batch size: {batch_size}')

        if split:
            _test_size = int(len(input_tensor) * split)
            _train_size = len(input_tensor) - _test_size
            _train, _test = random_split(input_tensor, (_train_size, _test_size))
            return DataLoader(_train, batch_size=batch_size, shuffle=True), DataLoader(_test, batch_size=batch_size, shuffle=False)
        else:
            return DataLoader(input_tensor, batch_size=batch_size, shuffle=False), None

    def _get_performance(self, data_loader, stats=None):
        if isinstance(data_loader, DataLoader):
            self.model.eval()

            predictions = []
            true_labels = []
            with torch.no_grad():
                for _, (input, label) in enumerate(data_loader):
                    out = self.model(input)
                    predictions.append(out)
                    true_labels.append(label)
            
            predictions = np.array(torch.concat(predictions, 0).tolist())
            true_labels = np.array(torch.concat(true_labels, 0).tolist())

            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            _pred = (predictions > .5).astype(int)
            acc = accuracy_score(true_labels, _pred)
            prec = precision_score(true_labels, _pred)
            rec = recall_score(true_labels, _pred)
            f1 = f1_score(true_labels, _pred)

            self.logger.info(f'accuracy: {acc * 100.:.2f}')
            self.logger.info(f'precision: {prec * 100.:.2f}')
            self.logger.info(f'recall: {rec * 100.:.2f}')
            self.logger.info(f'f1 score: {f1 * 100.:.2f}')

            if isinstance(stats, dict):
                stats['predictions'] = predictions
                stats['true_labels'] = true_labels

    def _train_phase(self, ep: int, data_loader, loss_hist=None):
        total_loss = .0
        self.model.train()

        if isinstance(data_loader, DataLoader):
            n_batches = len(data_loader)
            for _, (input, label) in enumerate(data_loader):
                out = self.model(input)
                loss = self.criterion(out, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        
            if isinstance(loss_hist, list):
                loss_hist.append([ep, total_loss / n_batches])

    def _test_phase(self, ep: int, data_loader, loss_hist=None):
        total_loss = .0
        self.model.eval()

        if isinstance(data_loader, DataLoader):
            n_batches = len(data_loader)
            with torch.no_grad():
                for _, (input, label) in enumerate(data_loader):
                    out = self.model(input)
                    loss = self.criterion(out, label)
                    total_loss += loss.item()

            if isinstance(loss_hist, list):
                loss_hist.append([ep, total_loss / n_batches])

    def _train_phase_gan(self, ep: int, data_loader, loss_hist=None):
        total_loss = total_loss_d = total_loss_d_real = total_loss_d_fake = .0

        self.model.train()
        self.model_d.train()

        if isinstance(data_loader, DataLoader):
            n_batches = len(data_loader)
            real_likelihood = self.params_d.get('real_likelihood') or 1.
            fake_likelihood = self.params_d.get('fake_likelihood') or .0
            real_weight = self.params_d.get('real_weight') or .5
            fake_weight = self.params_d.get('fake_weight') or 1. - real_weight

            for _, (input, label) in enumerate(data_loader):
                cur_size = input.shape[0]
                real_likelihoods = torch.full((cur_size, 1), real_likelihood, device=self.device)
                fake_likelihoods = torch.full((cur_size, 1), fake_likelihood, device=self.device)
                target_likelihoods = torch.ones((cur_size, 1), device=self.device)
                
                # Train Discriminator with real dataset
                likelihood = self.model_d(input, label)
                loss_d_real = self.criterion_d(likelihood, real_likelihoods)

                # Train Discriminator to identify generated dataset
                gen, gen_labels = self.model.generate_random(cur_size, self.device)
                likelihood = self.model_d(gen, gen_labels)
                loss_d_fake = self.criterion_d(likelihood, fake_likelihoods)

                loss_d = (loss_d_real * real_weight) + (loss_d_fake * fake_weight)
                self.optimizer_d.zero_grad()
                loss_d.backward()
                self.optimizer_d.step()

                # Train Generator to generate realistic dataset
                gen, gen_labels = self.model.generate_random(cur_size, self.device)
                likelihood = self.model_d(gen, gen_labels)
                loss = self.criterion(likelihood, target_likelihoods)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_loss_d += loss_d.item()
                total_loss_d_real += loss_d_real.item()
                total_loss_d_fake += loss_d_fake.item()

            if isinstance(loss_hist, list):
                loss_hist.append([ep, 
                    total_loss / n_batches, 
                    total_loss_d / n_batches, 
                    total_loss_d_real / n_batches, 
                    total_loss_d_fake / n_batches
                ])

    def _test_phase_gan(self, ep: int, data_loader, loss_hist=None):
        total_loss = total_loss_d = total_loss_d_real = total_loss_d_fake = .0
        self.model.eval()
        self.model_d.eval()

        if isinstance(data_loader, DataLoader):
            n_batches = len(data_loader)
            real_weight = self.params_d.get('real_weight') or .5
            fake_weight = self.params_d.get('fake_weight') or 1. - real_weight

            with torch.no_grad():
                for _, (input, label) in enumerate(data_loader):
                    cur_size = input.shape[0]
                    real_likelihoods = torch.ones((cur_size, 1), device=self.device)
                    fake_likelihoods = torch.zeros((cur_size, 1), device=self.device)
                    # Test Discriminator with real dataset
                    likelihood = self.model_d(input, label)
                    loss_d_real = self.criterion_d(likelihood, real_likelihoods)

                    # Test Discriminator with generated dataset
                    gen, gen_labels = self.model.generate_random(cur_size, self.device)
                    likelihood = self.model_d(gen, gen_labels)
                    loss_d_fake = self.criterion_d(likelihood, fake_likelihoods)
                    loss = self.criterion(likelihood, real_likelihoods)

                    loss_d = (loss_d_real * real_weight) + (loss_d_fake * fake_weight)
                    
                    total_loss += loss.item()
                    total_loss_d += loss_d.item()
                    total_loss_d_real += loss_d_real.item()
                    total_loss_d_fake += loss_d_fake.item()

            if isinstance(loss_hist, list):
                loss_hist.append([ep, 
                    total_loss / n_batches, 
                    total_loss_d / n_batches, 
                    total_loss_d_real / n_batches, 
                    total_loss_d_fake / n_batches
                ])

    def _train_proc(self, stats=None, **args):
        inputs = args.get('inputs')
        labels = args.get('labels')

        epochs = self.params.get('epochs') or 100
        block_size = self.params.get('block_size') or int(epochs / 10)

        test_split = self.params.get('test_split')
        train_loader, test_loader = self._input_to_dataloader(inputs, labels, test_split)
        train_loss_hist = []
        test_loss_hist = []

        for ep in range(1, epochs+1):
            if ep % block_size == 0 or ep == 1 or ep == epochs:
                self._train_phase(ep, train_loader, train_loss_hist)
                self._test_phase(ep, test_loader, test_loss_hist)
                self.logger.info(f'Epoch {ep}/{epochs} ---------------------')
                self.logger.info(f'Train Loss: {train_loss_hist[-1]}')
                self.logger.info(f'Test Loss: {test_loss_hist[-1]}')
            else:
                self._train_phase(ep, train_loader)

        self.logger.info('END ----------------------------\n')
        self._get_performance(test_loader, stats)

        if isinstance(stats, dict):
            stats['train_loss'] = train_loss_hist
            stats['test_loss'] = test_loss_hist

    def _predict_proc(self, stats=None, **args):
        inputs = args.get('inputs')
        labels = args.get('labels')

        data_loader, _ = self._input_to_dataloader(inputs, labels)
        self.model.eval()

        predictions = []
        with torch.no_grad():
            for _, (input, label) in enumerate(data_loader):
                out = self.model(input)
                predictions.append(out)
        
        predictions = np.array(torch.concat(predictions, 0).tolist())

        acc = np.sum((predictions > .5) == labels) / len(labels) * 100.
        self.logger.info(f'avg accuracy: {acc:.2f}')

        if isinstance(stats, dict):
            stats['predictions'] = predictions
            stats['true_labels'] = labels
        return predictions

    def _predict_proc_gan(self, stats=None, **args):
        _args = args.get('args')
        d_threshold = _args.get('d_threshold') or .5 # data making discriminator output higher than this value
        safe_break = _args.get('safe_break') or 1000
        amounts = _args.get('amounts') or {}

        self.model.eval()
        self.model_d.eval()

        data = []
        labels = []
        for cls, amount in amounts.items():
            self.logger.info(f'generating {amount} samples for class {cls}')

            remaining = amount
            loop_count = 0
            while remaining > 0 and loop_count < safe_break:
                gen_labels = torch.full((remaining,), cls, device=self.device)
                out, _ = self.model.generate_random(remaining, self.device, gen_labels)
                likelihood = self.model_d(out, gen_labels)
                is_realistic = torch.flatten(likelihood >= d_threshold)
                array = out[is_realistic].detach().to(self.device).numpy()
                data.append(array)
                remaining -= sum(is_realistic)
                loop_count += 1
            
            labels.append(np.full((amount-remaining, 1), cls, dtype=int))
            if remaining > 0:
                self.logger.warning(f'failed to generate enough realistic dataset (class {cls}): {amount-remaining}/{amount}')
        
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        result = np.concatenate((data, labels), axis=1)

        return result