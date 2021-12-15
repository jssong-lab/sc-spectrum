#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 09:54:32 2021

@author: jake
"""

import numpy as np
import umap
from sklearn.manifold import TSNE


def tsne_visualization(
    embedding: np.ndarray,
    n_clust: int,
) -> np.ndarray:
    """
    Obtain a 2 dimensional t-SNE embedding to use for visualization.
    
    Arguments:
        embedding: the full dimensional embeddings for the cells
        n_clust: the number of clusters and hence embedding dimensions to use
        
    """
    
    emb_norm = embedding[:, :n_clust] / np.linalg.norm(embedding[:, :n_clust], axis=1, keepdims=True)
    
    emb_tsne = TSNE(n_components=2, init='pca').fit_transform(emb_norm)
    
    return emb_tsne


def umap_visualization(
    embedding: np.ndarray,
    n_clust: int,
    random_state: int = 12345678,
) -> np.ndarray:
    """
    Obtain a 2 dimensional UMAP embedding to use for visualization.
    
    Arguments:
        embedding: the full dimensional embeddings for the cells
        n_clust: the number of clusters and hence embedding dimensions to use
        
    """
    
    emb_norm = embedding[:, :n_clust] / np.linalg.norm(embedding[:, :n_clust],
                        axis=1, keepdims=True)
    
    mapper = umap.UMAP(random_state = random_state, n_neighbors = 30,
                   set_op_mix_ratio = 1, local_connectivity = 1,
                   min_dist = 0.3, metric="cosine")
    
    mapper.fit(emb_norm)
    emb_umap = mapper.embedding_
    
    return emb_umap
