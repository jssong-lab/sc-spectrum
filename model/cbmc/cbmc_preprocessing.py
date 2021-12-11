#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 14:14:52 2021

@author: jake
"""

import numpy as np
import pandas as pd
from typing import List
from model.preprocessing import rna_pca, clr_transform 


def filter_species(
    df: pd.DataFrame,
    species: str,
    frac: float = 0.9,
) -> pd.DataFrame:
    """
    Remove cells and genes from a specified species using RNA counts.
    
    Arguments
        df : pandas DataFrame containing the count data. Rows should correspond
            to genes and columns to cells
        species : The species to be kept in the matrix. Either MOUSE or HUMAN 
            for the CBMC dataset
        frac : The fraction of counts in genes not from the specified species
        
    Returns
        species_df : A pandas DataFrame with the genes and cells from the
            specified species removed
            
    """
    
    spec_genes = [gene for gene in df.index if gene[:len(species) + 1] == species + "_"]
    
    frac_spec = df.loc[spec_genes].sum(axis = 0) / df.sum(axis = 0)
    
    spec_cells = df.columns[np.where(frac_spec > frac)[0]]
    
    species_df = df.loc[spec_genes, spec_cells]
    
    return species_df


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
    
    cbmc_rna_count = pd.read_csv(rna_file, header = 0, index_col = 0)
        
    cbmc_rna_human_count = filter_species(cbmc_rna_count, species = "HUMAN")
    del cbmc_rna_count
    
    cbmc_rna_pca = rna_pca(cbmc_rna_human_count, n_pc, random_state)
    
    return cbmc_rna_pca
    

def preprocess_adt(
    adt_file: str,
    human_cells: List[str],
) -> pd.DataFrame:
    """
    Perform centered log ratio (CLR) transform of the ADT count data.
    
    Arguments:
        adt_file : The file containing the ADT count matrix
        human_cells : A list of str specifying the human cells to be kept in 
            the transformed matrix
            
    Returns:
        cbmc_adt_human_clr : A pandas DataFrame with the CLR transformed ADT 
            profiles for the human cells. Rows correspond to cells and columns
            to ADT antibodies
               
    """
    
    cbmc_adt_count = pd.read_csv(adt_file, header = 0, index_col = 0)
    
    cbmc_adt_human_count = cbmc_adt_count.loc[:, human_cells]
    del cbmc_adt_count
    
    cbmc_adt_human_clr = clr_transform(cbmc_adt_human_count)
    
    return cbmc_adt_human_clr
    
    
    