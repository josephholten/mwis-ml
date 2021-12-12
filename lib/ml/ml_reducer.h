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
    MISConfig mis_config;
    BoosterHandle booster;

    graph_access original_graph, G;
    std::vector<bool> exists;

    std::vector<graph_access> reduced_graphs;
    std::vector<std::vector<NodeID>> reverse_mappings;
    std::vector<std::vector<NodeID>> forced;

public:
    explicit ml_reducer(graph_access& original_graph, MISConfig mis_config, const std::string& model_filepath = "../models/latest.model");
    ~ml_reducer() noexcept(false);

    void train_model();
    // void ml_reduce_old(graph_access& G, graph_access& R, std::vector<NodeID>& reverse_mapping);
    void build_reduced_graph(graph_access &R);
    NodeWeight ml_reduce();
    NodeWeight iterative_reduce();

    void save_model();

    void apply_solution();
};


#endif //MWIS_ML_ML_REDUCER_H
