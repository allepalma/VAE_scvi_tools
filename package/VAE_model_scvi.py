import logging
from typing import List, Optional

from anndata import AnnData
from scvi.data import setup_anndata
from scvi.model._utils import _init_library_size
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin, RNASeqMixin
from scvi.utils import setup_anndata_dsp

from package.VAE_module_scvi import VAE

logger = logging.getLogger(__name__)


class VAEModel(RNASeqMixin, VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """
    Standard scvi-tools model.
    Please use this skeleton to create new models.
    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~mypackage.MyModel.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        **model_kwargs,
    ):
        super(VAEModel, self).__init__(adata)

        # Initialize variables
        self.adata = adata
        self.scvi_setup_dict_ = adata.uns["_scvi"]
        self.summary_stats = self.scvi_setup_dict_["summary_stats"]
        # Setup dimensions
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_layers = n_layers

        # Compute the library mean and sd per batch
        library_log_means, library_log_vars = _init_library_size(
            adata, self.summary_stats["n_batch"]
        )

        # The module is the variational autoencoder implemented earlier
        self.module = VAE(
            n_input=self.summary_stats["n_vars"],
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            **model_kwargs,
        )
        self._model_summary_string = "Standard variational autoencoder"
        # necessary line to get params that will be used for saving/loading
        self.init_params_ = self._get_init_params(locals())

        logger.info("The model has been initialized")

    @staticmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        adata: AnnData,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        layer: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        copy: bool = False,
    ) -> Optional[AnnData]:
        """
        %(summary)s.
        Parameters
        ----------
        %(param_adata)s annotation data
        %(param_batch_key)s the key with batch indexes
        %(param_labels_key)s the key with observation labels (if any)
        %(param_layer)s layer key where to extract the data (if any)
        %(param_cat_cov_keys)s categorical covariate key
        %(param_cont_cov_keys)s continuous covariate key
        %(param_copy)s
        Returns
        -------
        %(returns)s
        """
        # Run function setup_anndata
        return setup_anndata(
            adata,
            batch_key=batch_key,
            labels_key=labels_key,
            layer=layer,
            categorical_covariate_keys=categorical_covariate_keys,
            continuous_covariate_keys=continuous_covariate_keys,
            copy=copy,
        )



