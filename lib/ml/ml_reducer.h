//
// Created by joseph on 10/16/21.
//

#ifndef MWIS_ML_ML_REDUCER_H
#define MWIS_ML_ML_REDUCER_H

#include <graph_access.h>
#include <mis_config.h>
#include <branch_and_reduce_algorithm.h>
#include "tools/safe_c_api.h"
#include "tools/MLConfig.h"

class ml_reducer {
private:

    const float q {};  // confidence niveau
    MISConfig mis_config;

    BoosterHandle booster;

public:
    explicit ml_reducer(MISConfig mis_config, float q);
    ~ml_reducer() noexcept(false);

    void train_model();
    void ml_reduce(graph_access& G, graph_access& R, std::vector<NodeID>& reverse_mapping);
    NodeWeight iterative_reduce(graph_access& G);

    void save_model();
};


#endif //MWIS_ML_ML_REDUCER_H
