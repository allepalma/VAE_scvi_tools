import numpy as np
import torch
import torch.nn.functional as F
from scvi import _CONSTANTS
from scvi.distributions import ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import DecoderSCVI, Encoder, one_hot
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

#torch.backends.cudnn.benchmark = True


class VAE(BaseModuleClass):
    """
    Implementation of a basic variational auto-encoder model.
    Parameters
    ----------
    n_input
        Number of input genes
    library_log_means
        1 x n_batch array of means of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    library_log_vars
        1 x n_batch array of variances of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    n_batch
        Number of batches, if 0, no batch correction is performed.
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    dropout_rate
        Dropout rate for neural networks
    latent_distribution
        The distribution of the latent space variables
    """

    def __init__(
        self,
        n_input: int,
        library_log_means: np.ndarray,
        library_log_vars: np.ndarray,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        latent_distribution="normal"
    ):
        super().__init__()
        # Setup the basic variables from the initialization
        self.n_latent = n_latent
        self.n_batch = n_batch
        self.latent_distribution = latent_distribution

        # Here, the library scalings have a mean and a variance. These are parameters with priors assigned to them
        self.register_buffer(
            "library_log_means", torch.from_numpy(library_log_means).float()
        )
        self.register_buffer(
            "library_log_vars", torch.from_numpy(library_log_vars).float()
        )

        # Additional parameters of the model
        self.px_r = torch.nn.Parameter(torch.randn(n_input))
        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(
            n_input,
            1,
            n_layers=1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        # decoder goes from n_latent-dimensional space to n_input-d data
        self.decoder = DecoderSCVI(
            n_latent,
            n_input,
            n_layers=n_layers,
            n_hidden=n_hidden,
        )

    def _get_inference_input(self, tensors, **kwargs):
        """
        Parse the dictionary to get appropriate arguments for inference
        :param tensors: a tensor output from the DataLoader containing the data
        :return: input_dict: the input dictionary for inference
        """
        # Extract the input x
        x = tensors[_CONSTANTS.X_KEY]
        # Wrap x into an input dictionary
        input_dict = dict(x=x)
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs, **kwargs):
        """
        Retrieve and organize the inputs of the generation
        :param tensors: a tensor output from the DataLoader containing the data
        :param inference_outputs: the result of the inference process
        :return: input_dict: the input of the decoder
        """
        z = inference_outputs["z"]  # Encoder output
        library = inference_outputs["library"]  # Encoded library output

        input_dict = {
            "z": z,
            "library": library,
        }
        return input_dict

    @auto_move_data
    def inference(self, x, n_samples=1):
        """
        High level inference method.
        Runs the inference (encoder) model.
        """
        # log the raw counts for numerical stability
        x_ = torch.log(1 + x)
        # get variational parameters via the encoder networks
        qz_m, qz_v, z = self.z_encoder(x_)
        ql_m, ql_v, library = self.l_encoder(x_)  # The latent library parameter is a single scaling value

        outputs = dict(z=z, qz_m=qz_m, qz_v=qz_v, ql_m=ql_m, ql_v=ql_v, library=library)
        return outputs

    @auto_move_data
    def generative(self, z, library, transform_batch = None):
        """Inputs sampled latent variables and runs a generative model on it based on the likelihood function"""
        # form the parameters of the ZINB likelihood:
        #   - px_scale returns the scaled gene expression of for all genes in a cell after reconstruction
        #   - px_rate is the rate of the zinb and is obtained taking exponent of px_scale*library
        #   - px_droput computes one value per feature to decide which of them to dropout
        #   - px_r log parameter of the likelihood
        px_scale, _, px_rate, px_dropout = self.decoder("gene", z, library)
        px_r = torch.exp(self.px_r)

        return dict(
            px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout
        )

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
        """
        Calculate the VAE loss for ELBO optimization
        :param tensors: anndata object containing the input raw counts
        :param inference_outputs:  Outputs of the encoding mechanism
        :param generative_outputs: Outputs of the decoding mechanism
        :param kl_weight: Additional custom kl divergence scale
        :return: The loss in a loss recording object
        """
        x = tensors[_CONSTANTS.X_KEY]
        # Parameters for the posterior
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        ql_m = inference_outputs["ql_m"]
        ql_v = inference_outputs["ql_v"]
        # Parameters of the likelihood
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]

        # Parameters to sample from the prior on the latent space
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)

        # KL divergence between the posterior and the prior for regularization
        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(
            dim=1
        )

        # The library size is per batch
        batch_index = tensors[_CONSTANTS.BATCH_KEY]
        n_batch = self.library_log_means.shape[1]
        # Simple way to select the right library log mean and log variance for the the actual scaling process
        local_library_log_means = F.linear(
            one_hot(batch_index, n_batch), self.library_log_means
        )
        local_library_log_vars = F.linear(
            one_hot(batch_index, n_batch), self.library_log_vars
        )

        # Again comput the KL divergence but this time assessing the distance from the library scaling prior
        kl_divergence_l = kl(
            Normal(ql_m, torch.sqrt(ql_v)),
            Normal(local_library_log_means, torch.sqrt(local_library_log_vars)),
        ).sum(dim=1)

        # Compute the negatve binomial likelihood with the parameters estimated from the generation
        # HOW TO? use the output of the generation to parametrize the negative binomial and check the probability of x
        # under it.
        reconst_loss = (
            -ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
            .log_prob(x)
            .sum(dim=-1)
        )

        # Compute a weighted KL divergence
        kl_local_for_warmup = kl_divergence_z
        kl_local_no_warmup = kl_divergence_l
        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        # Calculate the mean loss with both reconstruction and the weighted KL divergence
        loss = torch.mean(reconst_loss + weighted_kl_local)

        kl_local = dict(
            kl_divergence_l=kl_divergence_l, kl_divergence_z=kl_divergence_z
        )
        kl_global = torch.tensor(0.0)
        return LossRecorder(loss, reconst_loss, kl_local, kl_global)

    @torch.no_grad()
    def sample(
        self,
        tensors,
        n_samples=1,
        library_size=1,
    ) -> np.ndarray:
        r"""
        Generate observation samples from the posterior predictive distribution.
        The posterior predictive distribution is written as :math:`p(\hat{x} \mid x)`.
        Parameters
        ----------
        tensors
            Tensors dict
        n_samples
            Number of required samples for each cell
        library_size
            Library size to scale samples
        Returns
        -------
        x_new : :py:class:`torch.Tensor`
            tensor with shape (n_cells, n_genes, n_samples)
        """
        # Parametrize the negative binomial and sample from
        inference_kwargs = dict(n_samples=n_samples)
        _, generative_outputs, = self.forward(
            tensors,
            inference_kwargs=inference_kwargs,
            compute_loss=False,
        )

        px_r = generative_outputs["px_r"]
        px_rate = generative_outputs["px_rate"]
        px_dropout = generative_outputs["px_dropout"]

        dist = ZeroInflatedNegativeBinomial(
            mu=px_rate, theta=px_r, zi_logits=px_dropout
        )

        if n_samples > 1:
            exprs = dist.sample().permute(
                [1, 2, 0]
            )  # Shape : (n_cells_batch, n_genes, n_samples)
        else:
            exprs = dist.sample()

        return exprs.cpu()

