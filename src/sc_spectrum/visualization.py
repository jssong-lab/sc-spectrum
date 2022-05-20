#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 09:54:32 2021

@author: jake
"""

import numpy as np
import pandas as pd
import umap
import seaborn as sns
import matplotlib.pyplot as plt


def umap_embed(
    embedding: np.ndarray,
    n_clust: int,
    random_state: int = 12345678,
) -> np.ndarray:
    """
    Obtain a 2 dimensional UMAP embedding to use for visualization.
    
    Arguments:
        embedding: The full dimensional embeddings for the cells
        n_clust: The number of clusters and hence embedding dimensions to use
        
    """
    
    emb_norm = embedding[:, :n_clust] / np.linalg.norm(embedding[:, :n_clust],
                        axis=1, keepdims=True)
    
    mapper = umap.UMAP(random_state = random_state, n_neighbors = 30,
                   set_op_mix_ratio = 1, local_connectivity = 1,
                   min_dist = 0.3, metric="cosine")
    
    mapper.fit(emb_norm)
    emb_umap = mapper.embedding_
    
    return emb_umap


def umap_scatter(
    ax: plt.axes,
    reduction: pd.DataFrame,
    cluster_labels: np.ndarray,
    show_labels: bool = True,
    legend: bool = True,
) -> None:
    """
    Scatter plot using a two dimensional reduction. Points are colored by
    cluster.
    
    Arguments:
        ax: A matplotlib.pyplot.axes object. This axes is used for drawing the
            scatter plot and for the legend
        reduction: A pandas DataFrame storing a two dimensional reduction to
            use for the scatter plot. The zero column will be plotted along
            the x axis, and the one column will be plotted along the y axis
        cluster_labels: Cluster labels to use for coloring the points
        legend: Whether or not to plot a legend indicating the cluster colors
    
    Returns: 
        None
        
    """
    
    if(len(np.unique(cluster_labels)) <= 36):
        palette_0 = sns.husl_palette(n_colors=12, h=0.05, s=0.9, l=0.65)
        palette_1 = sns.husl_palette(n_colors=12, h=0.05, s=0.9, l=0.75)
        palette_2 = sns.husl_palette(n_colors=12, h=0.05, s=0.9, l=0.55)
        colors = palette_0 + palette_1 + palette_2
    else:
        colors = sns.husl_palette(n_colors=len(np.unique(cluster_labels)),
                                  h=0.05, s=0.9, l=0.65)
        
    labels = np.unique(cluster_labels)
    for i, label in enumerate(labels):
        barcodes = reduction.index[ (np.where(cluster_labels == label))[0] ]
        n_label = len(barcodes)
        rx = reduction.loc[barcodes].iloc[:, 0]
        ry = reduction.loc[barcodes].iloc[:, 1]
        ax.scatter(rx, ry, s=3, linewidths=0, alpha=1, color=colors[i], label=f"Clus. {(label):d}: ({n_label:d} cells)")
        if(show_labels):
            ax.text(rx.median(), ry.median(), (label), ha='center', va='center')
    
    if(legend):
        leg = ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.01), 
                        markerscale = 5, ncol = 1, fontsize = 11)
        for lh in leg.legendHandles: 
            lh.set_alpha(1)
            
            
def recolor_clusters(
    ref_cl: np.ndarray,
    map_cl: np.ndarray,
) -> np.ndarray:
    """
    A function to reorder the cluster labels in one clustering relative to a 
    different clustering. The two clustering results must have the same 
    number of clusters.
    
    Arguments:
        ref_cl: The reference clustering results
        map_cl: The clustering results to relabel to match the cluster labels 
            to the most similar clusters in the reference clustering result
        
    Returns:
        np.array([remap[c] for c in map_cl]): The clustering labels from map_cl
            reordered to match the labels to similar clusters in the reference
        
    """
    
    n_ref = len(np.unique(ref_cl))
    n_map = len(np.unique(map_cl))
    
    overlap = np.zeros((n_ref, n_map))
    remap = {}
    for i1, c1 in enumerate(np.unique(map_cl)):
        
        for i2, c2 in enumerate(np.unique(ref_cl)):
            for k in range(len(ref_cl)):
                if((ref_cl[k] == c1) and (map_cl[k] == c2)):
                    overlap[i1, i2] += 1
                    
        sorted_overlap = np.argsort(overlap[i1])
        step = 0
        while(len(remap) == i1):
            if(sorted_overlap[-step-1] not in remap.values()):
                remap[c1] = np.unique(ref_cl)[sorted_overlap[-step-1]]
            step += 1
       
    return np.array([remap[c] for c in map_cl])
    
