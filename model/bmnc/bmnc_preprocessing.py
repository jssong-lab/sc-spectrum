#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 15:33:12 2021

@author: jake
"""

import pandas as pd

from model.preprocessing import rna_pca, clr_transform 


def preprocess_rna(
    rna_file: str,
    n_pc: None,
    random_state: int,
) -> pd.DataFrame:
    """
    
    
    
    """
    
    rna_count = pd.read_csv(rna_file, header = 0, index_col = 0)
    
    bm_rna_pca = rna_pca(rna_count, n_pc, random_state)
    
    return bm_rna_pca


def preprocess_adt(
    adt_file: str,
) -> pd.DataFrame:
    """
    
    """
    
    bm_adt_count = pd.read_csv(adt_file, header = 0, index_col = 0)
  
    bm_adt_clr = clr_transform(bm_adt_count)  
    
    return bm_adt_clr