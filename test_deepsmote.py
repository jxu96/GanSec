import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scripts.data_loader import get_dataloader, get_dataset, get_windows
from scripts.eval_dist import calculate_fid_tabular, calculate_mmd_rbf
from models.dnn import Classifier
from scripts.clf import train_clf, evaluate_clf
from scipy.io import savemat

input_dim = 5 * 72  # 360 - dimension of flattened features
label_dim = 5       # dimension of labels
latent_dim = 64     # Example latent dimension (tune this)
hidden_dim = 128    # Example hidden layer dimension (tune this)

##############################################################################

class Args:
    lr=.0002
    epochs=100
    batch_size=256
    n_z=64
    window=5
    lr_clf=0.001
    epoch_num=20
    block_size=10

args = Args()


##############################################################################



## create encoder model and decoder model
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, latent_dim)
            # No activation on the final layer is common for latent space
        )

    def forward(self, x):
        # x is expected to be flattened: [batch_size, input_dim]
        z = self.fc_layers(x)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Tanh() # Or nn.Sigmoid(), or None, depending on data normalization
                      # Tanh outputs [-1, 1], Sigmoid outputs [0, 1]
        )

    def forward(self, z):
        # z is the latent vector: [batch_size, latent_dim]
        x_hat = self.fc_layers(z)
        return x_hat

##############################################################################
"""set models, loss functions"""
# control which parameters are frozen / free for optimization
def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


##############################################################################
"""functions to create SMOTE images"""

def G_SM(X, y,n_to_sample,cl):

    # determining the number of samples to generate
    #n_to_sample = 10 

    # fitting the model
    n_neigh = 5 + 1
    nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
    nn.fit(X)
    dist, ind = nn.kneighbors(X)

    # generating samples
    base_indices = np.random.choice(list(range(len(X))),n_to_sample)
    neighbor_indices = np.random.choice(list(range(1, n_neigh)),n_to_sample)

    X_base = X[base_indices]
    X_neighbor = X[ind[base_indices, neighbor_indices]]

    samples = X_base + np.multiply(np.random.rand(n_to_sample,1),
            X_neighbor - X_base)

    #use 10 as label because 0 to 9 real classes and 1 fake/smoted = 10
    return samples, [cl]*n_to_sample

#xsamp, ysamp = SM(xclass,yclass)

def augment(encoder, decoder, A, ratio):
    num_original_samples = A.shape[0]
    num_samples_to_generate = int(ratio * num_original_samples)

    features_flat_np = A.reshape(num_original_samples, -1)
    features_tensor = torch.Tensor(features_flat_np)

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        latent_original = encoder(features_tensor.to(device)).cpu().numpy() # Shape: [4000, latent_dim]

    # --- Modified SMOTE Generation ---
    n_neighbors = 6 # 5 neighbors + self

    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(latent_original)
    distances, indices = nn.kneighbors(latent_original)

    # Generate 16,000 new samples
    base_indices = np.random.choice(num_original_samples, num_samples_to_generate)
    # For each base sample, choose one of its k (e.g., 5) nearest neighbors randomly
    neighbor_indices = indices[base_indices, np.random.choice(n_neighbors - 1, num_samples_to_generate) + 1]

    latent_base = latent_original[base_indices]
    latent_neighbor = latent_original[neighbor_indices]

    # Interpolation step
    gap = np.random.rand(num_samples_to_generate, 1)
    latent_new = latent_base + gap * (latent_neighbor - latent_base) # Shape: [16000, latent_dim]

    # --- Decode new latent samples ---
    latent_new_tensor = torch.Tensor(latent_new).to(device)
    with torch.no_grad():
        features_new_scaled = decoder(latent_new_tensor).cpu().numpy() # Shape: [16000, 360]

    # --- Inverse Transform to original feature scale ---
    # features_new_flat = scaler.inverse_transform(features_new_scaled) # Shape: [16000, 360]

    # --- Assign Labels to New Samples ---
    # A common strategy: assign the label of the 'base' sample
    labels_new = A_label[base_indices] # Shape: [16000, 5]

    return features_new_scaled.reshape(-1, 5, 72), labels_new

###############################################################################

device = 'cuda' if torch.cuda.is_available() else 'cpu'
scaler = MinMaxScaler()
A, A_label = get_dataset('data/ue_jamming_detection/train.csv')
B, B_label = get_dataset('data/ue_jamming_detection/valid.csv')
A[A.columns] = scaler.fit_transform(A)
B[B.columns] = scaler.transform(B)

A, A_label = get_windows(A, A_label, args.window)
B, B_label = get_windows(B, B_label, args.window)
# A, A_label = A.to_numpy().reshape(-1, 72, 1), A_label.reshape(-1, 1)
# B, B_label = B.to_numpy().reshape(-1, 72, 1), B_label.reshape(-1, 1)

A_0 = A[np.apply_along_axis(np.mean, 1, A_label) == 0]
A_1 = A[np.apply_along_axis(np.mean, 1, A_label) == 1]

A_train_loader, A_test_loader = get_dataloader(
    A, A_label, device=device, batch_size=args.batch_size, train_test_split=.2)
A_loader, _ = get_dataloader(
    A, A_label, device=device, batch_size=args.batch_size)
B_loader, _ = get_dataloader(
    B, B_label, device=device, batch_size=args.batch_size)

encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=input_dim)

decoder = decoder.to(device)
encoder = encoder.to(device)

if os.path.exists('backups/deepsmote'):
    encoder.load_state_dict(torch.load('backups/deepsmote/enc.pth', weights_only=True, map_location=torch.device('cpu')))
    decoder.load_state_dict(torch.load('backups/deepsmote/dec.pth', weights_only=True, map_location=torch.device('cpu')))

else:
    os.makedirs('backups/deepsmote', exist_ok=True)
    #decoder loss function
    criterion = nn.MSELoss()
    criterion = criterion.to(device)

    batch_size = 100
    num_workers = 0

    best_loss = np.inf

    enc_optim = torch.optim.Adam(encoder.parameters(), lr = args.lr)
    dec_optim = torch.optim.Adam(decoder.parameters(), lr = args.lr)

    for epoch in range(args.epochs):
        # train for one epoch -- set nets to train mode
        encoder.train()
        decoder.train()

        train_loss = .0

        for features, labels in A_train_loader:
            enc_optim.zero_grad()
            dec_optim.zero_grad()
            
            features, labels = features.to(device), labels.to(device)
            features = torch.reshape(features, (-1, 360))
            # labels = torch.reshape(-1, labels)
            # labsn = labs.detach().cpu().numpy()
        
            latent_vecs = encoder(features)
            reconstructed_features = decoder(latent_vecs)

            loss = criterion(reconstructed_features, features)
            train_loss += loss.item()
            #print('xhat ', x_hat.size())
            #print(x_hat)
            #print('mse ',mse)
            loss.backward()
            enc_optim.step()
            dec_optim.step()
        
        #store the best encoder and decoder models
        #here, /crs5 is a reference to 5 way cross validation, but is not
        #necessary for illustration purposes
        if train_loss < best_loss:
            path_enc = 'backups/deepsmote/enc.pth'
            path_dec = 'backups/deepsmote/dec.pth'
            
            torch.save(encoder.state_dict(), path_enc)
            torch.save(decoder.state_dict(), path_dec)

            best_loss = train_loss  

if not os.path.exists('backups/deepsmote/eval.mat'):
    augment_ratios = [.1, .2, .3, .5, .7, 1., 1.5, 2., 3., 5.]

    outputs = {}

    for ratio in augment_ratios:
        ratio_pc = int(ratio*100)
        print('Augmenting with {}% ..'.format(ratio_pc))
        amount = int(A.shape[0] * ratio // 2)

        x_flag_0, y_flag_0 = augment(encoder, decoder, A_0, ratio/2)
        x_flag_1, y_flag_1 = augment(encoder, decoder, A_1, ratio/2)

        A_flag = np.concatenate([A, x_flag_0, x_flag_1], axis=0)
        A_label_flag = np.concatenate([A_label, y_flag_0, y_flag_1], axis=0)

        A_flag_train_loader, A_flag_test_loader = get_dataloader(A_flag, A_label_flag, device=device, batch_size=args.batch_size, train_test_split=.1)

        clf_augmented = Classifier(
            data_shape=[A.shape[1], A.shape[2]],
            label_shape=[5, 1],
        ).to(device)

        train_clf(clf_augmented, A_flag_train_loader, A_flag_test_loader, args)

        key_A = f'deepsmote_A_{ratio_pc}'
        key_B = f'deepsmote_B_{ratio_pc}'

        results_A = evaluate_clf(clf_augmented, A_loader, args)
        results_B = evaluate_clf(clf_augmented, B_loader, args)

        outputs[key_A] = results_A
        outputs[key_B] = results_B

        print('({}) Perf on A set: {}'.format(ratio_pc, results_A))
        print('({}) Perf on B set: {}'.format(ratio_pc, results_B))

    savemat('backups/deepsmote/eval.mat', outputs)

if not os.path.exists('backups/deepsmote/dist.mat'):
    results = {
        'deepsmote_0_fid': [],
        'deepsmote_0_mmd': [],
        'deepsmote_1_fid': [],
        'deepsmote_1_mmd': [],
    }
    eval_loop = 100

    for i in range(eval_loop):
        print(f'{i}/{eval_loop}')
        x_flag_0, _ = augment(encoder, decoder, A_0, 1.)
        results['deepsmote_0_fid'].append(calculate_fid_tabular(A_0, x_flag_0))
        results['deepsmote_0_mmd'].append(calculate_mmd_rbf(A_0, x_flag_0))

        x_flag_1, _ = augment(encoder, decoder, A_1, 1.)
        results['deepsmote_1_fid'].append(calculate_fid_tabular(A_1, x_flag_1))
        results['deepsmote_1_mmd'].append(calculate_mmd_rbf(A_1, x_flag_1))

    savemat(f'backups/deepsmote/dist.mat', results)
