#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 10:46:47 2021

@author: jake
"""

import argparse
import numpy as np
import torch

from bmnc_preprocessing import preprocess_rna, preprocess_adt

from model.scml import rbf_neighbor_graph, sparse_spectral, spectral_clustering
from model.scml import scml


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f_rna", "--file_rna", type = str, required = True,
                        help = ("The file containing the RNA count data.    \n"
                                "(GSE100866_CBMC_8K_13AB_10X-RNA_umi.csv.gz)\n"
                                "Can be downloaded from GSE100866. Required"))
    parser.add_argument("-f_adt", "--file_adt", type = str, required = True,
                        help = ("The file containing the ADT count data.    \n"
                                "(GSE100866_CBMC_8K_13AB_10X-ADT_umi.csv.gz)\n"
                                "Can be downloaded from GSE100866. Required"))
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
    
    bmnc_rna_pca = preprocess_rna(args.file_rna, args.n_pc, random_state)
    bmnc_adt_clr = preprocess_adt(args.file_adt)
    
    Gs_rna = rbf_neighbor_graph(bmnc_rna_pca.values.astype(np.float32), adaptive = True)
    L_rna, w_rna, v_rna = sparse_spectral(Gs_rna, n_clust = n_clust, random_state = random_state)
    cl_rna = spectral_clustering(v_rna, n_clust = n_clust, random_state = random_state)
    
    Gs_adt = rbf_neighbor_graph(bmnc_adt_clr.values.astype(np.float32), adaptive = True)
    L_adt, w_adt, v_adt = sparse_spectral(Gs_adt, n_clust = n_clust, random_state = random_state)
    cl_adt = spectral_clustering(v_adt, n_clust = n_clust, random_state = random_state)
    
    w_scml, v_scml, scml_cls = scml([Gs_rna, Gs_adt], n_clust = n_clust,
                                    device = device, alpha = alpha, random_state = random_state)
    
    np.save(outdir.rstrip("/") + "bmnc_v_scml_" + alpha, v_scml)
    np.save(outdir.rstrip("/") + "bmnc_w_scml_" + alpha, w_scml)
    np.save(outdir.rstrip("/") + "bmnc_cluster_labels_" + alpha, scml_cls)
    