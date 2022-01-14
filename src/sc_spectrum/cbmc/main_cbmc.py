#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 10:58:51 2021

@author: Jacob Leistico
"""

import argparse
import numpy as np
import pandas as pd
import torch

from cbmc_preprocessing import preprocess_rna, preprocess_adt

from sc_spectrum.scml import rbf_neighbor_graph, sparse_spectral, spectral_clustering
from sc_spectrum.scml import scml, soft_scml

from sc_spectrum.visualization import umap_embed

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from sklearn.metrics import silhouette_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f_rna", "--file_rna", type = str, required = True,
                        help = ("The file containing the RNA count data.    \n"
                                "(GSE100866_CBMC_8K_13AB_10X-RNA_umi.csv.gz)\n"
                                "Can be downloaded from GSE100866. Required"))
    parser.add_argument("-f_adt", "--file_adt", type = str, required = True,
                        help = ("The file containing the ADT count data.    \n"
                                "(GSE100866_CBMC_8K_13AB_10X-ADT_umi.csv.gz)\n"
                                "Can be downloaded from GSE100866. Required"))
    parser.add_argument("-cl", "--comp_file", type = str, default = None,
                        help = ("A file containing cluster labels to use for\n"
                                "comparing alternate methods"))
    parser.add_argument("--n_clust", type = int, required = True)
    parser.add_argument("--alpha", type = float, default = 100.)
    parser.add_argument("--n_pc", type = int, default = 30)
    parser.add_argument("--use_gpu", action = "store_true")
    parser.add_argument("--random_state", type = int, default = 12345678)
    parser.add_argument("-o", "--outdir", type = str, default = "./",
                        help = ("A path to a directory where the output     \n"
                                "files will be returned. By default the     \n"
                                "output files will be writen to the working \n"
                                "directory."))
    
    args = parser.parse_args()
    
    n_clust = args.n_clust
    alpha = args.alpha
    random_state = args.random_state
    outdir = args.outdir
    comp_file = args.comp_file
    
    use_gpu = args.use_gpu
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")    
    
    
    cbmc_rna_pca = preprocess_rna(args.file_rna, args.n_pc, random_state)
    cbmc_adt_clr = preprocess_adt(args.file_adt, list(cbmc_rna_pca.index))
    # cbmc_adt_clr = pd.read_csv(args.file_adt, header = 0, index_col = 0).T.loc[cbmc_rna_pca.index]
    
    Gs_rna = rbf_neighbor_graph(cbmc_rna_pca.values.astype(np.float32), adaptive = True)
    L_rna, w_rna, v_rna = sparse_spectral(Gs_rna, n_clust = n_clust, random_state = random_state)
    cl_rna = spectral_clustering(v_rna, n_clust = n_clust, random_state = random_state)
    
    Gs_adt = rbf_neighbor_graph(cbmc_adt_clr.values.astype(np.float32), adaptive = True)
    L_adt, w_adt, v_adt = sparse_spectral(Gs_adt, n_clust = n_clust, random_state = random_state)
    cl_adt = spectral_clustering(v_adt, n_clust = n_clust, random_state = random_state)
    
    w_scml, v_scml, cl_scml = scml([Gs_rna, Gs_adt], n_clust = n_clust,
                                   device = device, alpha = alpha,
                                   random_state = random_state)
    
    w_soft_scml, v_soft_scml, cl_soft_scml = soft_scml([Gs_rna, Gs_adt],
                                                       n_clust = n_clust,
                                                       device = device,
                                                       alpha = alpha,
                                                       random_state = random_state)
    
    cluster_methods = ["RNA spectral", "ADT spectral",
                       "SCML {}".format(alpha),
                       "soft-SCML {}".format(alpha)]
    cbmc_cls = pd.DataFrame(np.stack((cl_rna, cl_adt, cl_scml, cl_soft_scml)).T,
                            index = cbmc_rna_pca.index,
                            columns = cluster_methods)
    
    cbmc_cls.to_csv(outdir.rstrip("/") + "/cbmc_cluster_labels.csv",
                    header = True, index = True)
    
    
    D_adt = squareform(pdist(cbmc_adt_clr.values.astype(np.float32), metric = "euclidean"))
    D_rna = squareform(pdist(cbmc_rna_pca.values.astype(np.float32), metric = "euclidean"))
    
    # if(comp_file):
    #     comp_labels = pd.read_csv(comp_file, header = True, index = True).loc[cbmc_rna_pca.index]
    f = open(outdir.rstrip("/") + "/cbmc_clustering_metrics.csv", "w+")
    f.write("Clustering method,RNA distance silhouette,")
    f.write("ADT distance silhouette,Embedding distance silhouette\n")
    for labels, vecs, method in zip([cl_rna, cl_adt, cl_scml, cl_soft_scml],
                                    [v_rna, v_adt, v_scml, v_soft_scml],
                                    cluster_methods):
        vecs_norm = vecs[:, :n_clust] / np.linalg.norm(vecs[:, :n_clust], axis = 1, keepdims = True)
        f.write("{},".format(method))
        f.write("{},".format(silhouette_score(D_rna, labels, metric = "precomputed")))
        f.write("{},".format(silhouette_score(D_adt, labels, metric = "precomputed")))
        f.write("{}\n".format(silhouette_score(vecs_norm, labels, metric = "euclidean")))
        
    f.close()
            
    cbmc_umap = pd.DataFrame(umap_visualization(v_scml, n_clust),
                             index = cbmc_rna_pca.index,
                             columns = ["UMAP 1", "UMAP 2"])
    cbmc_umap.to_csv(outdir.rstrip("/") + "/cbmc_scml_umap.csv")
    