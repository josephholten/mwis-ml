//
// Created by joseph on 10/14/21.
//

#include <graph_access.h>
#include <graph_io.h>
#include <mis_config.h>
#include <branch_and_reduce_algorithm.h>

#include <xgboost/c_api.h>

#include "features.h"

#define safe_xgboost(call) {                                            \
int err = (call);                                                       \
if (err != 0) {                                                         \
    throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
                        ": error in " + #call  + ":" + XGBGetLastError()); \
    }                                                                    \
}


std::vector<std::string> split_by_lines(const std::string& file_name) {
    std::vector<std::string> lines;
    std::ifstream ifstream(file_name);
    std::string line;
    while (std::getline(ifstream, line)) {
        lines.push_back(line);
    }
    return lines;
}

int main(int argn, char** argv) {
    // mis_log::instance()->restart_total_timer();

    MISConfig mis_config;
    std::string train_graphs_list_path("train_graphs_list.txt");
    std::string test_graphs_list_path("test_graphs_list.txt");

    // get combined feature matrices for training and testing graphs
    std::vector<float> train_feat, test_feat;
    features_from_paths(mis_config, split_by_lines(train_graphs_list_path), train_feat);
    features_from_paths(mis_config, split_by_lines(train_graphs_list_path), test_feat);

    // make DMatrices
    DMatrixHandle dtrain, dtest;
    XGDMatrixCreateFromMat(&train_feat[0], train_feat.size(), FEATURE_NUM, 0, &dtrain);
    XGDMatrixCreateFromMat(&train_feat[0], train_feat.size(), FEATURE_NUM, 0, &dtrain);

    

    free(dtrain);
    free(dtest);


    // mis_config.graph_filename = graph_filepath.substr(graph_filepath.find_last_of( '/' ) +1);
    // mis_log::instance()->set_config(mis_config);


}
