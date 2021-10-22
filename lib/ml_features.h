//
// Created by joseph on 9/29/21.
//

#ifndef KAMIS_FEATURES_H
#define KAMIS_FEATURES_H

#include "graph_access.h"
#include "mis_config.h"

const int FEATURE_NUM = 13;
enum kw : int { NODES=0, EDGES=1, DEG=2, CHI2_DEG=3, AVG_CHI2_DEG=4, LCC=5, CHI2_LCC=6, CHROMATIC=7, T_WEIGHT=8, NODE_W=9, W_DEG=10, CHI2_W_DEG=11, LOCAL_SEARCH=12 };

void features(const MISConfig& mis_config, graph_access& G, std::vector<float>::iterator feat_mat);

#endif //KAMIS_FEATURES_H