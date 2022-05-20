#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 15:33:12 2021

@author: jake
"""

import pandas as pd

from sc_spectrum.preprocessing import rna_pca, clr_transform 

def preprocess_rna(
    rna_file: str,
    n_pc: None,
    random_state: int,
) -> pd.DataFrame:
    """
    Process the RNA count matrix for the CBMC dataset.
    
    Arguments
        rna_file : The file containing the RNA count matrix
        n_pc : How many pricipal components to compute. If None all principal
            components will be computed. If an integer is specieid this number
            of principal components will be return. If a float is given the
            number of principal components needed to explain this fraction of
            the variance will be returned
        random_state : The random state to use for the 'randomized' solver.
            This is only used when an integer value for n_pc is given
            
    Returns
        cbmc_rna_pca : A pandas DataFrame with the RNA principal components.
            Rows correspond to cells and columns to principal components
            
    """
    
    rna_count = pd.read_csv(rna_file, header = 0, index_col = 0)
    
    bm_rna_pca = rna_pca(rna_count, n_pc, random_state)
    
    return bm_rna_pca


def preprocess_adt(
    adt_file: str,
) -> pd.DataFrame:
    """
    Perform centered log ratio (CLR) transform of the ADT count data.
    
    Arguments:
        adt_file : The file containing the ADT count matrix
            
    Returns:
        cbmc_adt_human_clr : A pandas DataFrame with the CLR transformed ADT 
            profiles. Rows correspond to cells and columns to ADT antibodies.
            
    """
    
    bm_adt_count = pd.read_csv(adt_file, header = 0, index_col = 0)
  
    bm_adt_clr = clr_transform(bm_adt_count)  
    
    return bm_adt_clr