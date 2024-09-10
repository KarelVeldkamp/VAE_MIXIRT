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

# initialise model and optimizer
logger = CSVLogger("logs", name=cfg['which_data'], version=0)
trainer = Trainer(fast_dev_run=cfg['single_epoch_test_run'],
                  max_epochs=cfg['max_epochs'],
                  logger=logger,
                  callbacks=[EarlyStopping(monitor='train_loss', min_delta=cfg['min_delta'], patience=cfg['patience'], mode='min')])

if cfg['which_data'] == 'load':
    true_class = torch.squeeze(torch.Tensor(pd.read_csv('./data/class.csv', index_col=0).values)) -1
    true_theta = pd.read_csv('./data/theta.csv', index_col=0).values
    true_difficulty = pd.read_csv('./data/difficulty.csv', index_col=0).values
    dataset = CSVDataset('data/data.csv')
elif cfg['which_data'] == 'sim':
    # Step 1: Creating true_class tensor with torch
    true_class = np.expand_dims(np.random.binomial(1, cfg['class_prob'], cfg['N']), -1)
    covMat = np.full((cfg['mirt_dim'], cfg['mirt_dim']), 0)  # covariance matrix of dimensions, zero for now
    np.fill_diagonal(covMat, 1)
    true_theta = np.random.multivariate_normal([0] * cfg['mirt_dim'], covMat, cfg['N'])
    true_difficulty = np.random.uniform(-2, 2, (cfg['nitems'], 2))
    true_slopes = np.random.uniform(.5, 2, (cfg['nitems'],2))
    #true_slopes = np.ones((cfg['nitems'], 2))


    theta_repeat = np.repeat(true_theta, cfg['nitems'], -1)
    b0 = true_difficulty[:, 0]
    b1 = true_difficulty[:, 1]
    a0 = np.expand_dims(true_slopes[:, 0], -1)
    a1 = np.expand_dims(true_slopes[:, 1], -1)


    if cfg['mirt_dim'] >1:
        Q = pd.read_csv(f'./QMatrices/QMatrix{cfg["mirt_dim"]}D.csv', header=None).values
        a0 *= Q
        a1 *= Q
    else:
        Q = None


    exponent = (np.dot(true_theta, a0.T) + b0) * true_class + (np.dot(true_theta, a1.T) + b1) * (1-true_class)

    prob = np.exp(exponent) / (1 + np.exp(exponent))
    data = np.random.binomial(1, prob).astype(float)

    dataset = SimDataset(data)
    true_class = np.squeeze(true_class)







train_loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False)
vae = VAE(dataloader=train_loader,
          nitems=cfg['nitems'],
          learning_rate=cfg['learning_rate'],
          latent_dims=cfg['mirt_dim'],
          hidden_layer_size=50,
          qm=None,
          batch_size=5000,
          n_iw_samples=cfg['n_iw_samples'],
          temperature_decay=cfg['temperature_decay'],
          beta=1)

# vae.decoder.weights1.requires_grad_(False)
# vae.decoder.weights2.requires_grad_(False)
# vae.decoder.bias1.requires_grad_(False)
# vae.decoder.bias2.requires_grad_(False)

trainer.fit(vae)


# calculate predicted class labels
a1_est = vae.decoder.weights1.detach().cpu().numpy()
a2_est = vae.decoder.weights2.detach().cpu().numpy()
d1_est = vae.decoder.bias1.detach().cpu().numpy()
d2_est = vae.decoder.bias2.detach().cpu().numpy()


dataset = SimDataset(data)
train_loader = DataLoader(dataset, batch_size=data.shape[0], shuffle=False)
data = next(iter(train_loader))
_, log_sigma_est, cl = vae.encoder(data)
post_samples = vae.fscores(data)
theta = post_samples.mean(0)
print(vae.GumbelSoftmax(cl))
cl = torch.argmax(cl, dim=1)


# label switching
if pearsonr(cl, true_class).statistic < 0:
    print(1)
    # swap group labels
    tmp = cl.clone()  # Create a copy of the original vector
    cl[tmp == 0] = 1
    cl[tmp == 1] = 0

    # swap difficulty paramters
    d1_est, d2_est = d2_est, d1_est
    # swap slope parameters
    a1_est, a2_est = a2_est, a1_est



acc = torch.mean((cl== true_class).float())
print(f'Latent class accuracy: {acc.item():.4f}')

theta = torch.flatten(theta).detach().numpy()

plt.figure()
mse = MSE(theta, true_theta)
plt.scatter(y=theta, x=true_theta)
plt.plot(true_theta, true_theta)
plt.title(f'Theta:, MSE={round(mse,4)}')
plt.xlabel('True values')
plt.ylabel('Estimates')
plt.savefig(f'./figures/{cfg["which_data"]}/theta.png')



plt.figure()
mse = MSE(d1_est, true_difficulty[:, 1])
plt.scatter(y=d1_est, x=true_difficulty[:, 1])
plt.plot(true_difficulty[:, 1], true_difficulty[:, 1])
plt.title(f'Difficulty 1:, MSE={round(mse,4)}')
plt.xlabel('True values')
plt.ylabel('Estimates')
plt.savefig(f'./figures/{cfg["which_data"]}/difficulty_1.png')

plt.figure()
mse = MSE(d2_est, true_difficulty[:, 0])
plt.scatter(y=d2_est, x=true_difficulty[:,0])
plt.plot(true_difficulty[:, 0], true_difficulty[:, 0])
plt.title(f'Difficulty 2:, MSE={round(mse,4)}')
plt.xlabel('True values')
plt.ylabel('Estimates')
plt.savefig(f'./figures/{cfg["which_data"]}/difficulty_2.png')

plt.figure()
mse = MSE(a1_est, true_slopes[:, 1])
plt.scatter(y=a1_est, x=true_slopes[:, 1])
plt.plot(true_slopes[:, 1], true_slopes[:, 1])
plt.title(f'Slopes 1:, MSE={round(mse,4)}')
plt.xlabel('True values')
plt.ylabel('Estimates')
plt.savefig(f'./figures/{cfg["which_data"]}/slopes_1.png')

plt.figure()
mse = MSE(a2_est, true_slopes[:, 0])
plt.scatter(y=a2_est, x=true_slopes[:, 0])
plt.plot(true_slopes[:, 0], true_slopes[:, 0])
plt.title(f'Slopes 2:, MSE={round(mse,4)}')
plt.xlabel('True values')
plt.ylabel('Estimates')
plt.savefig(f'./figures/{cfg["which_data"]}/slopes_2.png')

# plot training loss
plt.figure()
logs = pd.read_csv(f'logs/{cfg["which_data"]}/version_0/metrics.csv')
plt.plot(logs['epoch'], logs['train_loss'])
plt.title('Training loss')
plt.savefig(f'./figures/training_loss.png')




