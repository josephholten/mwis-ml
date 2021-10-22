# mwis-ml

## Quick Start

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


