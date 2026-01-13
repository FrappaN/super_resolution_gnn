# super_resolution_gnn

Code repository for the submission of "Super-Resolution of Urban Socioeconomic Indicators via Graph-Based Recommender Systems" at the Web Applications and Smart Cities Workshop at WWW 2026. 

The code requirements are in the requirements.txt file. 

The yelp dataset must be downloaded from https://business.yelp.com/data/resources/open-dataset/ and then processed through the "Yelp_dataset_processing.ipynb" notebook.

The models can be re-trained by running:
```
python train.py
python train_lightgcn.py
python train_interpolation_classification.py --n_quantiles 2
python train_super-res_classification.py --n_quantiles 2
```
The first script trains the RGCN and Emb models, while the second script trains the LightGCN models (see paper for details). The embeddings are already available in the results/yelp2019 folder. 
The subsequent scripts perform the socioeconomic tasks presented in the paper; launching the script will automatically download the demographics from datacommons.

The analysis on the embedding of the clusters, presented in the appendix is instead in "clustering_business.ipynb".
