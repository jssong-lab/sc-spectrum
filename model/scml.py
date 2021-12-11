#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from typing import Tuple
from typing import List


def rbf_neighbor_graph(
    X: np.ndarray,
    adaptive: bool = True,
    k: int = 30,
    a: int = 10,
) -> csr_matrix:
    """
    Construct a symmetric nearest neighbor graph with weights obtained 
    using a Gaussian radial basis function (rbf) kernel.
    
    Arguments:
        X : np.ndarray of shape n_cells by n_features to obtain pairwise
            distances from
        adaptive : bool specifying whether to use an adaptive or global width
            in the rbf kernel
        k : int specifying the number of neighbors to use in the graph
        a : int specifying how to set the width for the rbf. When adaptive is
            True this parameter sets the width to the distance to the ath
            nearest neighbor. When adaptive is False the width is set to the 
            ath percentile of nearest neighbor distances.
            
    Returns: 
        0.5 * (G + G.T) : The symmetric adjacency graph stored as a csr_matrix
        
    """
    
    kNN = NearestNeighbors(n_neighbors = k, metric="minkowski", p = 2)
    kNN.fit(X)
    G = kNN.kneighbors_graph(mode = "distance")
    
    if(adaptive):
        sigma = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            sigma[i] = np.sort(G.getrow(i).data)[a - 1]
        indptr = G.indptr
        for i in range(X.shape[0]):
            G.data[indptr[i]: indptr[i + 1]] = np.exp( - (G.data[indptr[i]: indptr[i + 1]] / sigma[i]) ** 2)
        
    else:
        sigma = np.percentile(G.data, a)
        G.data = np.exp(- (G.data / sigma) ** 2)
    
    return 0.5 * (G + G.T)
    
    
def reorder_clusters(
    cl: np.array
) -> np.array:
    """
    Reorder cluster labels by size
    
    Arguments:
        cl : Numpy array containing cluster labels for each cell
        
    Returns:
        cl_reorder : Integer cluster labels for the cells ordered
            by decreasing cluster size
            
    """
    
    clust_size = {}
    clust_size = {c:0 for c in np.unique(cl)}
    for label in cl:
        clust_size[label] += 1
    
    clust_sort = sorted(clust_size.items(), key = lambda x: x[1], reverse = True) 
    conversion_dict = {label[0]: i for i, label in enumerate(clust_sort)}
    
    cl_reorder = cl.copy()
    for i, label in enumerate(cl):
        cl_reorder[i] = conversion_dict[label]
    
    return cl_reorder


def sparse_spectral(
    A: csr_matrix,
    n_clust: int,
    random_state: int = 12345678,
) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
    """
    Spectral decomposition for a sparse adjacency matrix A.
    
    Arguments:
        A : The sparse adjacency matrix
        n_clust : the number of eigenvalues and corresponding eigenvectors
            to return
            
    Returns:
        L : The symmetric graph Laplacian
        eig_vals : The first n_clust eigenvalues of the symmetric graph 
            Laplacian
        eig_vecs : The first n_clust eigenvectors of the symmetric graph 
            Laplacian
    
    """
    v0 = np.random.default_rng(random_state).uniform(-1, 1, size = A.shape[0])
    
    D = diags(np.array(A.sum(axis=1))[:,0]).tocsr()
    D1_2 = D.sqrt().power(-1)
    
    L = D1_2 * (D - A) * D1_2
    
    eig_vals, eig_vecs = eigsh(L, k=n_clust, which="SM", v0 = v0)
    
    return L, eig_vals, eig_vecs


def spectral_clustering(
    eig_vecs: np.ndarray,
    n_clust: int,
    random_state: int = 12345678,
) -> np.ndarray:
    """
    Perform k-means clustering given an eigenvector embedding.
    
    Arguments:
        eig_vecs : The eigenvector embedding matrix. Each column gives an
            eigenvector sorted in ascending order by eigenvalue
        n_clust : The number of clusters to partition the data. This also 
            specifies the number of eigenvectors to use for the embedding.
        random_state : A random state to use for initializing the k-means
            search
            
    Returns:
        cluster_labels : The cluster labels for each cell. Each cluster is 
            assigned an integer label and these labels are sorted in descending
            order by cluster size
            
    """
    
    vecs_norm = eig_vecs[:, :n_clust] / np.linalg.norm(eig_vecs[:, :n_clust], axis = 1, keepdims = True)
    
    cluster_labels = KMeans(n_clusters = n_clust,
                        max_iter = 1000,
                        n_init = 100,
                        random_state = random_state).fit_predict(vecs_norm)
    
    return reorder_clusters(cluster_labels)


def scml(
    graphs: List[csr_matrix],
    n_clust: int,
    device: torch.device,
    alpha: float = 0.1,
    random_state: int = 12345678,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform the spectral clustering on multilayer graphs for a list of graph
    adjacency matrices.
    
    Arguments:
        graphs : A list of sparse csr_matrix corresponding to the graph
            adjacency matrices.
        n_clust : The number of clusters to use for clustering
        device : A PyTorch device to use for obtaining the nearest neighbors
        alpha : A weight to use for the penalty term for the distances to the
            individual subspaces
            
    Returns:
        vals_scml : The eigenvalues of the scml matrix
        vecs_scml : The eigenvectors of the scml matrix (as the columns)
        cluster_labels : The scml cluster labels for each cell
    
    """
    
    L, eig_vals, eig_vecs = sparse_spectral(graphs[0], n_clust, random_state)
    L_scml = L.toarray() - alpha * (eig_vecs @ eig_vecs.T)
    
    for G in graphs[1:]:
        L, eig_vals, eig_vecs = sparse_spectral(G, n_clust, random_state)
        L_scml += (L.toarray() - alpha * (eig_vecs @ eig_vecs.T))
        
    L_scml = torch.from_numpy(L_scml).to(device)
    vals_scml, vecs_scml = torch.linalg.eigh(L_scml)
    
    vals_scml = vals_scml.cpu().numpy()
    vecs_scml = vecs_scml.cpu().numpy()
    
    cluster_labels = spectral_clustering(vecs_scml, n_clust, random_state)
    
    return vals_scml, vecs_scml, cluster_labels


def full_spectral(
    A: csr_matrix,
    device: torch.device,
) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """
    Spectral decomposition for a sparse adjacency matrix A.
    
    Arguments:
        A : The sparse adjacency matrix
        n_clust : The number of eigenvalues and corresponding eigenvectors
            to return
    Returns:
        L : The symmetric graph Laplacian stored as a PyTorch tensor object
        eig_vals : The eigenvalues of the symmetric graph Laplacian
        eig_vecs : The eigenvectors of the symmetric graph Laplacian
    
    """
    
    D = diags(np.array(A.sum(axis=1))[:,0]).tocsr()
    D1_2 = D.sqrt().power(-1)
    
    L = D1_2 * (D - A) * D1_2
    L = torch.from_numpy(L.toarray()).to(device)
    
    eig_vals, eig_vecs = torch.linalg.eigh(L)
    
    return L, eig_vals, eig_vecs
 

def density_matrix(
    eig_vals: torch.Tensor,
    eig_vecs: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """
    Construct the density matrix from the spectrum and corresponding
    eigenvectors of the symmetric graph Laplacian.
    
    Arguments:
        eig_vals : Eigenvalues of the symmetric graph Laplacian
        eig_vecs : Eigenvectors of the symmetric graph Laplacian
        beta : Inverse temperature parameter. Larger values will suppress
            the contribution of eigenvector directions with larger eigenvalues
    
    Returns:
        rho / Z : The density matrix
        
    """

    P = torch.exp(-beta * eig_vals)
    rho = eig_vecs * P
    rho = rho @ eig_vecs.T
    Z = torch.trace(rho)
    
    return rho / Z

    
def soft_scml(
    graphs: List[csr_matrix],
    n_clust: int,
    device: torch.device,
    alpha: float = 0.1,
    beta: float = None,
    random_state: int = 12345678,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform the soft scml embedding and clustering.
    
    Arguments:
        graphs : A list of sparse csr_matrix corresponding to the graph
            adjacency matrices.
        n_clust : The number of clusters to use for clustering
        device : A PyTorch device to use for obtaining the nearest neighbors
        alpha : A weight to use for the penalty term for the distances to the
            individual subspaces
        beta : Inverse temperature parameter. Larger values will suppress
            the contribution of eigenvector directions with larger eigenvalues
        random_state : A random state to use for initializing the k-means
            search
    
    Returns:
        vals_scml : The eigenvalues of the soft scml matrix
        vecs_scml : The eigenvectors of the soft scml matrix (as the columns)
        cluster_labels : The soft scml cluster labels for each cell
        
    """
    
    N = graphs[0].shape[0]
    if beta is None:
        beta = np.sqrt(N)
        
    L_scml, eig_vals, eig_vecs = full_spectral(graphs[0], device)
    rho = density_matrix(eig_vals, eig_vecs, beta)
    del eig_vecs, eig_vals
    L_scml -= alpha * rho
    del rho
    
    for G in graphs[1:]:
        L, eig_vals, eig_vecs = full_spectral(G, device)
        L_scml += L
        del L
        rho = density_matrix(eig_vals, eig_vecs, beta)
        del eig_vecs, eig_vals
        L_scml -= alpha * rho
        del rho
           
    vals_scml, vecs_scml = torch.linalg.eigh(L_scml)
    del L_scml
    vals_scml = vals_scml.cpu().numpy()
    vecs_scml = vecs_scml.cpu().numpy()
    
    cluster_labels = spectral_clustering(vecs_scml, n_clust, random_state)
    
    return vals_scml, vecs_scml, cluster_labels



        
        