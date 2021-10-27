//
// Created by joseph on 10/27/21.
//

#ifndef MWIS_ML_FEATURE_MATRIX_H
#define MWIS_ML_FEATURE_MATRIX_H

#include <vector>

#include "graph_access.h"

class feature_matrix {
public:
    explicit feature_matrix(graph_access &g);
    int getNumberOfFeatures();

    void reserveGraph(graph_access& G);   // for reserving memory
    void fillGraph(graph_access& G, const std::vector<float>& labels);

private:
    const int FEATURE_NUM = 13;
    enum feature : int { NODES=0, EDGES=1, DEG=2, CHI2_DEG=3, AVG_CHI2_DEG=4, LCC=5, CHI2_LCC=6, CHROMATIC=7, T_WEIGHT=8, NODE_W=9, W_DEG=10, CHI2_W_DEG=11, LOCAL_SEARCH=12 };

    graph_access& G;

    template<feature f>
    void setFeature(NodeID node, float value);

    std::vector<float> feature_data;
    std::vector<float> label_data;

    size_t size;

};


#endif //MWIS_ML_FEATURE_MATRIX_H
