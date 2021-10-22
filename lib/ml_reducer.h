//
// Created by joseph on 10/16/21.
//

#ifndef MWIS_ML_ML_REDUCER_H
#define MWIS_ML_ML_REDUCER_H

#include <graph_access.h>
#include <mis_config.h>
#include <branch_and_reduce_algorithm.h>

class ml_reducer {
private:
    const float q;  // confidence niveau
    graph_access& G; // graph
    const MISConfig mis_config;

public:
    explicit ml_reducer(MISConfig mis_config, graph_access& G, float q);

    void ml_reduce(graph_access& R, std::vector<NodeID>& reverse_mapping);
    void iterative_reduce();
};


#endif //MWIS_ML_ML_REDUCER_H
