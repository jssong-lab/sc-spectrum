#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 11:43:37 2021

@author: Jacob Leistico
"""

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

def library_size_norm(
    X: np.ndarray,        
) -> np.ndarray:
    """
    Library size normalization for scRNA-seq UMI matrix.
    
    Arguments:
        X : UMI count matrix. The rows should correspond to genes and columns
            should correspond to cells
    Returns:
        X_norm : Normalized expression matrix. Count data is scaled per cell by
            library size and then square root transformed
            
    """
    
    libsize = X.sum(axis = 0)
    X_norm = X / libsize
    X_norm *= np.median(libsize)
    X_norm = np.sqrt(X_norm)
    
    return X_norm


def rna_pca(
    rna_count: pd.DataFrame,
    n_pc: None,
    random_state: int = 12345678,
) -> pd.DataFrame:
    """
    Library size normalization followed by projection onto the top principal
    components.
    
    Arguments:
        rna_count : pandas DataFrame with the scRNA UMI count data. Rows should
            correspond to genes and columns to cells
        n_pc : If an integer this specifies the number of principal components
            to return. If a float < 1.0 then the number of principal components
            necessary to explain that fraction of the variance are returned. If
            None all principal components are returned
        random_state: The random state to use for the 'randomized' solver.
            This is only used when an integer value for n_pc is given
        
    Returns:
        rna_pca : pandas DataFrame with the principal component projections of
            the scRNA UMI count data. Rows correspond to cells and columns to
            principal components
        
    """
    
    rna_norm = library_size_norm(rna_count.values)
    
    rna_pca = PCA(n_components = n_pc, random_state = random_state).fit_transform(rna_norm.T)
    
    rna_pca = pd.DataFrame(rna_pca, index = rna_count.columns,
                           columns = ["PC" + str(i) for i in range(rna_pca.shape[1])])
    
    return rna_pca 
    
    
def clr_transform(
    df: pd.DataFrame,
    pseudo: float = 1.,
) -> pd.DataFrame:
    """
    Centered log ratio (CLR) transform for ADT count data.
    
    Arguments:
        df : pandas DataFrame with the ADT count data. Rows should correspond 
            to ADT antibodies and columns should correspond to cells
        pseudo : Pseudo count to add to each element of the ADT count matrix
    
    Returns:
        df_clr : CLR transformed ADT count data. Rows should correspond to 
            cells and columns to ADT antibodies
            
    """
    
    g = np.prod(df.values + pseudo, axis = 0) ** (1. / df.shape[0])
    
    X_clr = np.log((df.values + pseudo) / g)
    
    df_clr = pd.DataFrame(X_clr.T,
                          index = df.columns,
                          columns = df.index)
        
    return df_clr