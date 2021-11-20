//
// Created by joseph on 9/29/21.
//

#ifndef KAMIS_WEIGHTED_LS_H
#define KAMIS_WEIGHTED_LS_H

// void initial_is(graph_access& G);
bool is_IS(graph_access& G);
// void perform_ils(const MISConfig& mis_config, graph_access& G, NodeWeight weight_offset);
// NodeWeight extractReductionOffset(const std::string & comments);

class weighted_ls {
private:
    MISConfig mis_config;
    graph_access& G;

public:
    weighted_ls(MISConfig mis_config, graph_access& G);
    void run_ils();
};


#endif //KAMIS_WEIGHTED_LS_H
