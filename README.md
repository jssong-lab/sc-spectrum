# sc-spectrum

Single-Cell Spectral analysis Using Multilayer graphs (sc-spectrum) is a package for clustering 
cells in multi-omic single-cell sequencing datasets. The package provides an 
implementation of the Spectral Clustering on Multilayer graphs (SCML) algorithm 
[[1]](#1) for the application to multi-omic single-cell sequencing datasets. The 
package provides functions for obtaining similarity graphs for each of the 
cellular modalities and for performing single modality spectral clustering or
the SCML clustering.


## Installation
To install the package, open a terminal and navigate to a directory where the package 
can be downloaded. Then clone the repository from github

    git clone https://github.com/jssong-lab/sc-spectrum.git

Move into the downloaded repo folder

    cd sc-spectrum

and create and acticate a virtual environment

    python3 -m venv venv
    source venv/bin/activate
    
For reproducible results upgrade pip and install the package requirements

    pip install --upgrade pip
    pip install -r requirements.txt
    
Alternatively the `requirements.txt` file can be generated with sub-dependencies
that are more specific to your operating system using `pip-tools`

    pip install --upgrade pip
    pip install pip-tools
    pip-compile requirements.in
    pip install -r requirements.txt
    
The sc-spectrum package can be built and installed by running

    pip install wheel
    sh build-package.sh
    sh install-built-package.sh

Alternatively the sc-spectrum package can be installed in development mode by running

    sh install-devmode.sh
    
### Reproducing SCML clustering results for the cbmc dataset
The SCML clustering results can be reproduced by running 

    python3 src/sc_spectrum/cbmc/main_cbmc.py -f_rna data_path/GSE100866_CBMC_8K_13AB_10X-RNA_umi.csv \
                                              -f_adt data_path/GSE100866_CBMC_8K_13AB_10X-ADT_umi.csv \
                                              --n_clust 12 -o outdir

where `data_path` is a path to where the input UMI count data can be found
and `outdir` is a directory where output files can be written

A demonstration of how the sc-spectrum package can be used for a general dataset and
how the figures were generated can be found in [notebooks/cbmc_analysis.ipynb](notebooks/cbmc_analysis.ipynb)

## References
<a id="1">[1]</a> 
Dong, X. et al. (2013). 
Clustering on multi-layer graphs via subspace
analysis on Grassmann manifolds. 
*IEEE Transactions on Signal
Processing*, 62(4), 905–918.
