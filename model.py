
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.prune
import pytorch_lightning as pl


class GumbelSoftmax(pl.LightningModule):
    def __init__(self):
        super(GumbelSoftmax, self).__init__()
        # Gumbel distribution
        self.G = torch.distributions.Gumbel(0, 1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.temperature = 10
        self.temperature_decay = .9999

    def forward(self, log_pi):
        # sample gumbel variable and move to correct device
        g = self.G.sample(log_pi.shape)
        g = g.to(log_pi)
        # sample from gumbel softmax
        y = self.softmax((log_pi + g)/self.temperature)
        return y


class Encoder(pl.LightningModule):
    """
    Neural network used as encoder
    """
    def __init__(self,
                 nitems: int,
                 ndim: int,
                 hidden_layer_size: int):
        """
        Initialisation
        :param latent_dims: number of latent dimensions of the model
        """
        super(Encoder, self).__init__()

        self.dense1 = nn.Linear(nitems, hidden_layer_size)
        self.dense2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.densem = nn.Linear(hidden_layer_size, ndim)
        self.denses = nn.Linear(hidden_layer_size, ndim)
        self.densec = nn.Linear(hidden_layer_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        A forward pass though the encoder network
        :param x: a tensor representing a batch of response data
        :param m: a mask representing which data is missing
        :return: a sample from the latent dimensions
        """

        # calculate s and mu based on encoder weights
        out = F.elu(self.dense1(x))
        out = F.elu(self.dense2(out))
        mu = self.densem(out)
        log_sigma = self.denses(out)
        cl = self.densec(out)

        return mu, log_sigma, cl


class Decoder(pl.LightningModule):
    """
    Neural network used as decoder
    """

    def __init__(self, nitems: int, latent_dims: int,  qm: torch.Tensor=None):
        """
        Initialisation
        :param latent_dims: the number of latent factors
        :param qm: IxD Q-matrix specifying which items i<I load on which dimensions d<D
        """
        super().__init__()
        # one layer for each class
        self.linear1 = nn.Linear(latent_dims, nitems, bias=True)
        self.linear1.weight = nn.Parameter(torch.ones(self.linear1.weight.shape), requires_grad=False)
        self.linear2 = nn.Linear(latent_dims, nitems, bias=True)
        self.linear2.weight = nn.Parameter(torch.ones(self.linear2.weight.shape), requires_grad=False)
        self.activation = nn.Sigmoid()

        # remove edges between latent dimensions and items that have a zero in the Q-matrix
        if qm is not None:
            msk_wts = torch.ones((nitems, latent_dims), dtype=torch.float32)
            for row in range(qm.shape[0]):
                for col in range(qm.shape[1]):
                    if qm[row, col] == 0:
                        msk_wts[row][col] = 0
            torch.nn.utils.prune.custom_from_mask(self.linear1, name='weight', mask=msk_wts)
            torch.nn.utils.prune.custom_from_mask(self.linear2, name='weight', mask=msk_wts)

    def forward(self, cl: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Forward pass though the network
        :param x: tensor representing a sample from the latent dimensions
        :param m: mask representing which data is missing
        :return: tensor representing reconstructed item responses
        """
        #print(cl[:, 0:1])
        out = self.linear1(theta) * cl[:,0:1] + self.linear2(theta) * cl[:,1:2]
        out = self.activation(out)
        return out


class SamplingLayer(pl.LightningModule):
    def __init__(self):
        super(SamplingLayer, self).__init__()
        self.N = torch.distributions.Normal(0, 1)

    def forward(self, mu, log_sigma):
        sigma = torch.exp(log_sigma)
        error = self.N.sample(mu.shape)
        # potentially move error vector to GPU
        error = error.to(mu)
        return mu + sigma * error


class VAE(pl.LightningModule):
    """
    Neural network for the entire variational autoencoder
    """
    def __init__(self,
                 dataloader,
                 nitems: int,
                 latent_dims: int,
                 hidden_layer_size: int,
                 qm: torch.Tensor,
                 learning_rate: float,
                 batch_size: int,
                 beta: int = 1):
        """
        Initialisaiton
        :param latent_dims: number of latent dimensions
        :param qm: IxD Q-matrix specifying which items i<I load on which dimensions d<D
        """
        super(VAE, self).__init__()
        #self.automatic_optimization = False
        self.nitems = nitems
        self.dataloader = dataloader

        self.encoder = Encoder(nitems,
                               latent_dims,
                               hidden_layer_size
        )

        self.GumbelSoftmax = GumbelSoftmax()
        self.sampler = SamplingLayer()


        self.decoder = Decoder(nitems, latent_dims, qm)

        self.lr = learning_rate
        self.batch_size = batch_size
        self.beta = beta
        self.kl=0

    def forward(self, x: torch.Tensor, m: torch.Tensor=None):
        """
        forward pass though the entire network
        :param x: tensor representing response data
        :param m: mask representing which data is missing
        :return: tensor representing a reconstruction of the input response data
        """
        mu, log_sigma, cl = self.encoder(x)
        cl = self.GumbelSoftmax(cl)
        theta = self.sampler(mu, log_sigma)

        reco = self.decoder(cl, theta)

        # Calcualte the estimated probabilities

        # calculate kl divergence
        kl = 1 + 2 * log_sigma - torch.square(mu) - torch.exp(2 * log_sigma)
        kl = torch.sum(kl, dim=-1)
        self.kl = -.5 * torch.mean(kl)
        return reco

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)

    def training_step(self, batch, batch_idx):
        # forward pass

        data = batch
        X_hat = self(data)

        bce = torch.nn.functional.binary_cross_entropy(X_hat, batch, reduction='none')
        bce = torch.mean(bce) * self.nitems

        loss = bce + self.beta * torch.sum(self.kl)
        self.GumbelSoftmax.temperature *= self.GumbelSoftmax.temperature_decay
        self.log('train_loss',loss)

        return {'loss': loss}

    def train_dataloader(self):
        return self.dataloader