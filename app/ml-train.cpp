//
// Created by joseph on 10/14/21.
//
#include <fstream>
#include "safe_c_api.h"

#include "mis_config.h"
#include "graph_access.h"

#include "ml_features.h"
#include "io_wrapper.h"







int main(int argn, char** argv) {

    // mis_log::instance()->restart_total_timer();

    MISConfig mis_config;

    std::string train_graphs_path("../train_graphs_list.txt");
    std::string train_labels_path("../train_labels_list.txt");
    std::string test_graph_path("../test_graphs_list.txt");
    std::string test_label_path("../test_labels_list.txt");

    // get combined feature matrices for training and testing graphs
    std::vector<float> train_feat, train_labels, test_feat, test_labels;
    features_from_paths(mis_config, split_file_by_lines(train_graphs_path), train_feat);
    features_from_paths(mis_config, split_file_by_lines(test_graph_path), test_feat);

    // make DMatrices
    DMatrixHandle dtrain, dtest;
    XGDMatrixCreateFromMat(&train_feat[0], train_feat.size(), FEATURE_NUM, 0, &dtrain);
    XGDMatrixCreateFromMat(&test_feat[0], train_feat.size(), FEATURE_NUM, 0, &dtest);

    // set labels used in training
    train_labels.resize(train_feat.size() / FEATURE_NUM);  // number of rows of feature matrix
    test_labels.resize(test_feat.size() / FEATURE_NUM);  // number of rows of feature matrix
    labels_from_paths(split_file_by_lines(train_graphs_path), train_labels.begin());
    labels_from_paths(split_file_by_lines(test_graph_path), test_labels.begin());

    XGDMatrixSetFloatInfo(dtrain, "label", &train_labels[0], train_labels.size());
    XGDMatrixSetFloatInfo(dtest, "label", &test_labels[0], test_labels.size());

    // create booster
    BoosterHandle booster;
    DMatrixHandle eval_dmats[2] = {dtrain, dtest};
    safe_xgboost(XGBoosterCreate(eval_dmats, 2, &booster));

    // parameters
    // no gpu  // TODO: change training to gpu?
    safe_xgboost(XGBoosterSetParam(booster, "gpu_id", "-1"));

    safe_xgboost(XGBoosterSetParam(booster, "objective", "binary:logistic"));
    safe_xgboost(XGBoosterSetParam(booster, "eval_metric", "logloss"));

    safe_xgboost(XGBoosterSetParam(booster, "gamma", "0.1"));
    safe_xgboost(XGBoosterSetParam(booster, "max_depth", "3"));
    safe_xgboost(XGBoosterSetParam(booster, "eta", "1"));
    safe_xgboost(XGBoosterSetParam(booster, "verbosity", "1"));

    int n_trees = 10;
    const char *eval_names[2] = {"train", "test"};
    const char *eval_result = nullptr;
    for (int i = 0; i < n_trees; ++i) {
        safe_xgboost(XGBoosterUpdateOneIter(booster, i, dtrain));
        safe_xgboost(XGBoosterEvalOneIter(booster, i, eval_dmats, eval_names, 2, &eval_result));
        printf("%s\n", eval_result);
    }

    bst_ulong num_feature = 0;
    safe_xgboost(XGBoosterGetNumFeature(booster, &num_feature));
    printf("num_feature: %lu\n", (unsigned long) (num_feature));

    // if (mis_config.save_model) {
    time_t now = time(nullptr);
    struct tm* t = localtime(&now);
    char time_stamp_name[50];
    strftime(time_stamp_name, sizeof(time_stamp_name), "../models/%y-%m-%d_%H-%M-%S.model", t);
    safe_xgboost(XGBoosterSaveModel(booster, "../models/latest.model"));
    safe_xgboost(XGBoosterSaveModel(booster, time_stamp_name));
    // }

    // if (mis_config.predict_test) {

    // }

    // freeing
    safe_xgboost(XGBoosterFree(booster));
    safe_xgboost(XGDMatrixFree(dtrain));
    safe_xgboost(XGDMatrixFree(dtest));

}
