import torch
from torch.utils.data import DataLoader
from model import *
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data import *
from scipy.stats import pearsonr
import time
import os
import sys

def sigmoid(z):
    return 1/(1 + np.exp(-z))
def MSE(est, true):
    """
    Mean square error
    Parameters
    ----------
    est: estimated parameter values
    true: true paremters values

    Returns
    -------
    the MSE
    """
    return np.mean(np.power(est-true,2))

def Cor(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))

def Cor(x, y):
    """Correlate each n with each m.

    Parameters
    ----------
    x : np.array
      Shape N X T.

    y : np.array
      Shape M X T.

    Returns
    -------
    np.array
      N X M array in which each element is a correlation coefficient.

    """
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must ' +
                         'have the same number of timepoints.')
    s_x = x.std(1, ddof=n - 1)
    s_y = y.std(1, ddof=n - 1)
    cov = np.dot(x,
                 y.T) - n * np.dot(mu_x[:, np.newaxis],
                                  mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])


with open("./config.yml", "r") as f:
    cfg = yaml.safe_load(f)
    cfg = cfg['configs']

if len(sys.argv) > 1:
    cfg["mirt_dim"] = int(sys.argv[1])
    cfg['cov'] = float(sys.argv[2])
    cfg["n_iw_samples"] = int(sys.argv[3])
    cfg['learning_rate'] = float(sys.argv[4])
    iteration = int(sys.argv[5])
else:
    iteration = 1


# initialise model and optimizer
logger = CSVLogger("logs", name=cfg['which_data'], version=0)
trainer = Trainer(fast_dev_run=cfg['single_epoch_test_run'],
                  max_epochs=cfg['max_epochs'],
                  logger=logger,
                  callbacks=[EarlyStopping(monitor='train_loss', min_delta=cfg['min_delta'], patience=cfg['patience'], mode='min')])

if cfg['which_data'] == 'load':
    true_class = pd.read_csv(f'./true/pars/class_{cfg["mirt_dim"]}_{cfg["cov"]}.csv').values.astype('int')
    true_theta = pd.read_csv(f'./true/pars/theta_{cfg["mirt_dim"]}_{cfg["cov"]}.csv').values.astype('float')
    true_difficulty = pd.read_csv(f'./true/pars/difficulty_{cfg["mirt_dim"]}_{cfg["cov"]}.csv').values.astype('float')
    true_slopes0 = pd.read_csv(f'./true/pars/slopes0_{cfg["mirt_dim"]}_{cfg["cov"]}.csv').values.astype('float')
    true_slopes1 = pd.read_csv(f'./true/pars/slopes1_{cfg["mirt_dim"]}_{cfg["cov"]}.csv').values.astype('float')
    true_slopes = np.concatenate((np.expand_dims(true_slopes0,-1), np.expand_dims(true_slopes1,-1)), -1) # repeat for both classes

    Q = true_slopes[:,:, 0] != 0
    #true_class = np.squeeze(true_class)

    data =  pd.read_csv(f'./true/data/data_{cfg["mirt_dim"]}_{cfg["cov"]}_{iteration}.csv').values.astype('float')
elif cfg['which_data'] == 'sim':
    # Step 1: Creating true_class tensor with torch
    true_class = np.expand_dims(np.random.binomial(1, cfg['class_prob'], cfg['N']), -1)
    covMat = np.full((cfg['mirt_dim'], cfg['mirt_dim']), cfg['cov'])  # covariance matrix of dimensions, zero for now
    np.fill_diagonal(covMat, 1)
    true_theta = np.random.multivariate_normal([0] * cfg['mirt_dim'], covMat, cfg['N'])
    true_difficulty = np.random.uniform(-2, 2, (cfg['nitems'], 2))
    #true_slopes = np.random.uniform(.5, 2, (cfg['nitems'], cfg['mirt_dim'],2))
    true_slopes = np.repeat(np.random.uniform(.5, 2, (cfg['nitems'], cfg['mirt_dim'], 1)), 2, -1)

    b0 = true_difficulty[:, 0]
    b1 = true_difficulty[:, 1]
    a0 = true_slopes[:, :, 0]
    a1 = true_slopes[:, :, 1]


    if cfg['mirt_dim'] >1:
        Q = pd.read_csv(f'./QMatrices/QMatrix{cfg["mirt_dim"]}DSimple.csv', header=None).values.astype(float)
        a0 *= Q
        a1 *= Q
    else:
        Q = None



    exponent = (np.dot(true_theta, a0.T) + b0) * (1-true_class) + (np.dot(true_theta, a1.T) + b1) * (true_class)

    prob = np.exp(exponent) / (1 + np.exp(exponent))
    data = np.random.binomial(1, prob).astype(float)
    true_class = np.squeeze(true_class)



dataset = SimDataset(data)
train_loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False)
vae = VAE(dataloader=train_loader,
          nitems=data.shape[1],
          learning_rate=cfg['learning_rate'],
          latent_dims=cfg['mirt_dim'],
          hidden_layer_size=50,
          qm=Q,
          batch_size=5000,
          n_iw_samples=cfg['n_iw_samples'],
          temperature_decay=cfg['temperature_decay'],
          beta=1)

# vae.decoder.weights1.requires_grad_(False)
# vae.decoder.weights2.requires_grad_(False)
# vae.decoder.bias1.requires_grad_(False)
# vae.decoder.bias2.requires_grad_(False)

t1 = time.time()
trainer.fit(vae)
runtime = time.time() - t1
print(f'runtime: {runtime}')

# calculate predicted class labels
a1_est = vae.decoder.weights1.detach().cpu().numpy().copy()
a2_est = vae.decoder.weights1.detach().cpu().numpy().copy()
d1_est = vae.decoder.bias1.detach().cpu().numpy()
d2_est = vae.decoder.bias2.detach().cpu().numpy()
#indices = torch.Tensor([1,2,3,4,5,11,12,13,14,15, 21,22, 23,24,25]).int()
#d2_est = d1_est.copy()
#d2_est[indices] = d2_est_all[indices]

dataset = SimDataset(data)
train_loader = DataLoader(dataset, batch_size=data.shape[0], shuffle=False)
data = next(iter(train_loader))
_, log_sigma_est, cl = vae.encoder(data)
z_samples, cl_samples = vae.fscores(data)
theta = z_samples.mean(0).detach().numpy()
cl = cl_samples.mean(0)
cl = torch.argmax(cl, dim=1)

print(cl)
print(cl.float().mean())

# label switching
true_class = true_class.squeeze()

print(pearsonr(cl, true_class).statistic)


if pearsonr(cl, true_class).statistic < 0:
    # swap group labels
    tmp = cl.clone()  # Create a copy of the original vector
    cl[tmp == 0] = 1
    cl[tmp == 1] = 0

    # swap difficulty paramters
    d1_est, d2_est = d2_est, d1_est
    # swap slope parameters
    a1_est, a2_est = a2_est, a1_est

# inverting scales:
for dim in range(theta.shape[1]):
    print(pearsonr(true_theta[:,dim], theta[:,dim]).statistic)
    if pearsonr(true_theta[:,dim], theta[:,dim]).statistic < 0:
        theta[:,dim] *= -1
        a1_est[dim, :] *= -1
        a2_est[dim, :] *= -1



acc = torch.mean((cl== true_class).float())
est_slopes = np.transpose(np.concatenate((a1_est[:, :, np.newaxis], a2_est[:, :, np.newaxis]), -1), axes=(1,0,2))

est_difficulty = np.concatenate((d1_est[:, np.newaxis], d2_est[:, np.newaxis]),-1)
#est_difficulty = torch.concat((d1_est[:, np.newaxis], d2_est[:, np.newaxis]),-1).detach().numpy()

print(est_slopes.shape)
print(est_difficulty.shape)
print(true_slopes.shape)
print(true_difficulty.shape)

msea = np.mean((est_slopes[true_slopes!=0] - true_slopes[true_slopes!=0])**2)
msetheta = np.mean((theta-true_theta)**2)
msed = np.mean((true_difficulty-est_difficulty)**2)



if len(sys.argv) > 1:
    metrics =  [msea, msetheta, msed, acc, runtime]
    with open(f"results/metrics/vae_{'_'.join(sys.argv[1:])}.txt", 'w') as f:
        for metric in metrics:
            f.write(f"{metric}\n")
else:
    print(f'Latent class accuracy: {acc.item():.4f}')
    print(f'MSE(a): {msea.item():.4f}')
    print(f'MSE(b): {msed.item():.4f}')
    print(f'MSE(theta): {msetheta.item():.4f}')

    for filename in os.listdir(f'./figures/{cfg["which_data"]}/'):
        file_path = os.path.join(f'./figures/{cfg["which_data"]}/', filename)
        os.remove(file_path)
    for dim in range(cfg['mirt_dim']):
        plt.figure()
        mse = MSE(theta[:,dim], true_theta[:,dim])
        plt.scatter(y=theta[:,dim], x=true_theta[:,dim])
        plt.plot(true_theta[:,dim], true_theta[:,dim])
        plt.title(f'Theta:, MSE={round(mse,4)}')
        plt.xlabel('True values')
        plt.ylabel('Estimates')
        plt.savefig(f'./figures/{cfg["which_data"]}/theta_d{dim}.png')

        for cl in range(2):
            plt.figure()
            mse = MSE(est_slopes[:, dim, cl], true_slopes[:, dim, cl])
            plt.scatter(y=est_slopes[:, dim, cl], x=true_slopes[:, dim, cl])
            plt.plot(true_slopes[:, dim, cl], true_slopes[:, dim, cl])
            plt.title(f'Slopes {cl+1}:, MSE={round(mse, 4)}')
            plt.xlabel('True values')
            plt.ylabel('Estimates')
            plt.savefig(f'./figures/{cfg["which_data"]}/slopes_class{cl+1}_d{dim}.png')


    for cl in range(2):
        plt.figure()
        mse = MSE(est_difficulty[:, cl], true_difficulty[:, cl])
        plt.scatter(y=est_difficulty[:, cl], x=true_difficulty[:, cl])
        plt.plot(true_difficulty[:, cl], true_difficulty[:, cl])
        plt.title(f'Difficulty 1:, MSE={round(mse,4)}')
        plt.xlabel('True values')
        plt.ylabel('Estimates')
        plt.savefig(f'./figures/{cfg["which_data"]}/difficulty_class{cl}.png')



    # plot training loss
    plt.figure()
    logs = pd.read_csv(f'logs/{cfg["which_data"]}/version_0/metrics.csv')
    plt.plot(logs['epoch'], logs['train_loss'])
    plt.title('Training loss')
    plt.savefig(f'./figures/training_loss.png')




