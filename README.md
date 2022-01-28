# VAE_scvi_tools 

Implementation of a basic variational autoencoder (VAE) using scVI-tools. 

* `data` contains the downloaded version of a dataset of heart cells from Litviňuková, M., Talavera-López, C., Maatz, H., Reichart, D., Worth, C. L., Lindberg, E. L., … & Teichmann, S. A. (2020). Cells of the adult human heart. Nature, 588(7838), 466-472.
* `figures` contains umap plots of latent cell signature embeddings. It should be noted that the simple VAE does not perform batch correction
* `package` contains the VI module and model
* `trained_model` contains the weights of a model trained for 400 epochs
* `run_example.py` can be executed to reproduce the results