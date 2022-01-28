import scanpy as sc
from package.VAE_model_scvi import *
import scvi


# Import scvi heart data
adata = scvi.data.heart_cell_atlas_subsampled()

# Filter genes by frequency of expression in cells
sc.pp.filter_genes(adata, min_counts=3)


# Add a copy of the raw counts to the layer key
adata.layers["counts"] = adata.X.copy()  # preserve counts
sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize
sc.pp.log1p(adata)  # Take the logarithm of the counts
adata.raw = adata  # freeze the state in `.raw`

# Perform feature selection only keeping the highest variable genes
sc.pp.highly_variable_genes(
    adata,
    n_top_genes=1200,
    subset=True,
    layer="counts",
    flavor="seurat_v3",
    batch_key="cell_source"
)

# Setup the adata file for model training
VAEModel.setup_anndata(
    adata,
    layer="counts"
)

# Initialize the model for training
#model = VAEModel(adata)

# Train the model
#model.train(max_epochs=400)  # uncomment for retraining 

#model.save("./trained_model")
model = VAEModel.load("trained_model/", adata)


# Obtain the latent outputs of the model and store in the adata object
latent = model.get_latent_representation()
adata.obsm["X_scVI"] = latent


# run PCA then generate UMAP plots
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, use_rep="X_scVI")
sc.tl.umap(adata, min_dist=0.3)

sc.pl.umap(
    adata,
    color=["cell_type"],
    frameon=False,
    save='celltype.png'
)

sc.pl.umap(
    adata,
    color=["donor", "cell_source"],
    ncols=2,
    frameon=False,
    save='batch.png'
)


# Run differential expression analysis
de_df = model.differential_expression(
    groupby="cell_type",
    group1="Endothelial",
    group2="Fibroblast"
)
de_df.head()












