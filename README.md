# mwis-ml

## Quick Start

### TODO

- make own config struct and parser
- make feature matrix class, implemented just using vector and iterator
    * should also be able to create single matrix for multiple graphs
- merge training into ml_reduce class
- good logging (with seperate timing - only in debug mode)

### Prerequisites

- gcc, cmake
- conda (or mini conda)
- conda enviornment to install xgboost into, named ENV_NAME

### Installing

```console
git clone --recurse-submodules ...
cd mwis-ml/extern/xgboost
mkdir build
cd build
conda activate ENV_NAME
cmake -- -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
make install
```

### Building

```console
mkdir build
cd build
conda activate ENV_NAME
cmake .. -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
make
```


