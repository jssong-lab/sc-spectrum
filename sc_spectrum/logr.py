#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 14:05:46 2021

@author: jake
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from scipy.stats import chi2

from typing import Dict


def LOGR(
    X: np.ndarray,
    cl: np.ndarray,
    cutoff: float,
) -> Dict[int, np.ndarray]:
    """
    Implementation of LOGR scRNA-seq differential expression method.
    
    Arguments:
        X : count matrix with rows corresponding to cells and columns
            to features
        cl : clutser labels for the cells
        cutoff : p-value cutoff for differential expression
    
    Returns:
        
    """
    
    clust_markers = {}
    
    for k in np.unique(cl):
        membership = np.zeros(X.shape[0], dtype=np.int)
        membership[np.where(cl == k)[0]] = 1
        N1 = len(np.where(cl == k)[0])
        N0 = X.shape[0] - N1
        
        llr_pval = np.ones(X.shape[1])
        for j in range(X.shape[1]):
            logr = LogisticRegression()
            logr.fit(X[:,j].reshape(-1, 1), membership)
            pred = np.array(logr.predict_proba(X[:,j].reshape(-1, 1))[:,1])
            gene_score = log_loss(membership, pred)
            
            llf = - gene_score * (N1 + N0)
            llnull = N1 * np.log(N1 / (N1 + N0))
            llnull += N0 * np.log(N0 / (N1 + N0))
            llr=llf-llnull
            
            if(logr.coef_[0] > 0.0):
                llr_pval[j] = 1 - chi2.cdf(2 * llr, 1)                
            
        clust_markers[k] = llr_pval
        
    return clust_markers
            