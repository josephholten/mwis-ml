//
// Created by joseph on 10/27/21.
//

#include "feature_matrix.h"

feature_matrix::feature_matrix(graph_access &g) : G(g) {

}

int feature_matrix::getNumberOfFeatures() {
    return 0;
}

void feature_matrix::reserveGraph(graph_access &G) {

}

void feature_matrix::fillGraph(graph_access &G, const std::vector<float> &labels) {

}

template<feature_matrix::feature f>
void feature_matrix::setFeature(NodeID node, float value) {
    feature_data[node + f] = value;
}
