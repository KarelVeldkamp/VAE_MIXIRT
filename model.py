
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.prune
import pytorch_lightning as pl


class GumbelSoftmax(pl.LightningModule):
    def __init__(self, temperature_decay):
        super(GumbelSoftmax, self).__init__()
        # Gumbel distribution
        self.G = torch.distributions.Gumbel(0, 1)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.temperature = 1
        self.temperature_decay = temperature_decay

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
        out = self.linear1(theta) * cl[:,:,0:1] + self.linear2(theta) * cl[:,:,1:2]
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
                 n_iw_samples: int,
                 temperature_decay:int,
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

        self.GumbelSoftmax = GumbelSoftmax(temperature_decay)
        self.sampler = SamplingLayer()
        self.latent_dims = latent_dims


        self.decoder = Decoder(nitems, latent_dims, qm)

        self.lr = learning_rate
        self.batch_size = batch_size
        self.beta = beta
        self.n_samples = n_iw_samples
        self.kl=0

    def forward(self, x: torch.Tensor, m: torch.Tensor=None):
        """
        forward pass though the entire network
        :param x: tensor representing response data
        :param m: mask representing which data is missing
        :return: tensor representing a reconstruction of the input response data
        """
        mu, log_sigma, cl = self.encoder(x)
        mu = mu.repeat(self.n_samples,1,1)
        log_sigma = log_sigma.repeat(self.n_samples,1,1)
        cl = cl.repeat(self.n_samples,1,1)

        cl = self.GumbelSoftmax(cl)
        z = self.sampler(mu, log_sigma)

        reco = self.decoder(cl, z)

        return reco, mu, log_sigma, z

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)

    def training_step(self, batch, batch_idx):
        # forward pass
        data = batch
        reco, mu, log_sigma, z = self(data)

        mask = torch.ones_like(data)
        loss, _ = self.loss(data, reco, mask, mu, log_sigma, z)
        self.GumbelSoftmax.temperature *= self.GumbelSoftmax.temperature_decay
        self.log('train_loss',loss)

        return {'loss': loss}

    def train_dataloader(self):
        return self.dataloader

    def loss(self, input, reco, mask, mu, sigma, z):
        #calculate log likelihood

        input = input.unsqueeze(0).repeat(reco.shape[0], 1, 1) # repeat input k times (to match reco size)
        log_p_x_theta = ((input * reco).clamp(1e-7).log() + ((1 - input) * (1 - reco)).clamp(1e-7).log()) # compute log ll
        logll = (log_p_x_theta * mask).sum(dim=-1, keepdim=True) # set elements based on missing data to zero
        #
        # calculate KL divergence
        log_q_theta_x = torch.distributions.Normal(mu, sigma.exp()).log_prob(z).sum(dim = -1, keepdim = True) # log q(Theta|X)
        log_p_theta = torch.distributions.Normal(torch.zeros_like(z).to(input), scale=torch.ones(mu.shape[2]).to(input)).log_prob(z).sum(dim = -1, keepdim = True) # log p(Theta)
        kl =  log_q_theta_x - log_p_theta # kl divergence

        # combine into ELBO
        elbo = logll - kl
        # # perform importance weighting
        with torch.no_grad():
            weight = (elbo - elbo.logsumexp(dim=0)).exp()
        #
        loss = (-weight * elbo).sum(0).mean()


        return loss, weight

    def fscores(self, batch, n_mc_samples=50):
        data = batch

        if self.n_samples == 1:
            mu, _, _ = self.encoder(data)
            return mu.unsqueeze(0)
        else:
            scores = torch.empty((n_mc_samples, data.shape[0], self.latent_dims))
            for i in range(n_mc_samples):
                reco, mu, log_sigma, z = self(data)
                mask = torch.ones_like(data)
                loss, weight = self.loss(data, reco, mask, mu, log_sigma, z)

                idxs = torch.distributions.Categorical(probs=weight.permute(1,2,0)).sample()

                # Reshape idxs to match the dimensions required by gather
                # Ensure idxs is of the correct type
                idxs = idxs.long()

                # Expand idxs to match the dimensions required for gather
                idxs_expanded = idxs.unsqueeze(-1).expand(-1, -1, z.size(2))  # Shape [10000, 1, 3]

                # Use gather to select the appropriate elements from z
                output = torch.gather(z.transpose(0, 1), 1, idxs_expanded).squeeze().detach() # Shape [10000, latent dims]
                if self.latent_dims == 1:
                    output = output.unsqueeze(-1)

                scores[i, :, :] = output

            return scores