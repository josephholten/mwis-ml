//
// Created by joseph on 10/16/21.
//

#include <numeric>
#include "ml_reducer.h"

#include "tools/safe_c_api.h"

#include <fstream>
#include <wmis_interface/weighted_ls.h>
#include <graph_io.h>

#include "mis_config.h"
#include "graph_access.h"

#include "tools/io_wrapper.h"
#include "ml_features.h"

ml_reducer::ml_reducer(MISConfig _mis_config, const float _q, const std::string& model_filepath /* = "../models/latest.model" */) : mis_config {std::move(_mis_config)}, q {_q} {
    // init booster
    safe_xgboost(XGBoosterCreate(nullptr, 0, &booster));
    safe_xgboost(XGBoosterSetParam(booster, "eta", "1"));
    safe_xgboost(XGBoosterSetParam(booster, "nthread", "16"));

    safe_xgboost(XGBoosterLoadModel(booster, model_filepath.c_str()));

    bst_ulong num_of_features;
    safe_xgboost(XGBoosterGetNumFeature(booster, &num_of_features));
    assert(num_of_features == ml_features::FEATURE_NUM);
}

ml_reducer::~ml_reducer() noexcept(false) {
    safe_xgboost(XGBoosterFree(booster));
}

void ml_reducer::train_model() {
    std::string train_graphs_path("../train_graphs_list.txt");
    std::string train_labels_path("../train_labels_list.txt");
    std::string test_graph_path("../test_graphs_list.txt");
    std::string test_label_path("../test_labels_list.txt");

    // get combined feature matrices for training and testing graphs
    ml_features train_features(mis_config), test_features(mis_config);
    std::cout << "LOG: ml-train: getting feature matrices\n";

    train_features.fromPaths(split_file_by_lines(train_graphs_path),
                             split_file_by_lines(train_labels_path));
    test_features.fromPaths(split_file_by_lines(test_graph_path),
                             split_file_by_lines(test_label_path));

    // initialize DMatrices in with feature and label data
    train_features.initDMatrix();
    test_features.initDMatrix();

    // create booster
    DMatrixHandle eval_dmats[2] = {train_features.getDMatrix(), test_features.getDMatrix()};
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

    std::cout << "LOG: ml-train: starting training\n";
    int n_trees = 10;
    const char *eval_names[2] = {"train", "test"};
    const char *eval_result = nullptr;

    for (int i = 0; i < n_trees; ++i) {
        std::cout << "LOG: ml-train: round " << i << "of training\n";
        safe_xgboost(XGBoosterUpdateOneIter(booster, i, train_features.getDMatrix()));
        safe_xgboost(XGBoosterEvalOneIter(booster, i, eval_dmats, eval_names, 2, &eval_result));
        printf("%s\n", eval_result);
    }

    bst_ulong num_feature = 0;
    safe_xgboost(XGBoosterGetNumFeature(booster, &num_feature));
    assert((unsigned long) num_feature == (unsigned long) train_features.getNumberOfFeatures());

    // if (mis_config.predict_test) {

    // }
}

void ml_reducer::save_model() {
    // create timestamp name
    time_t now = time(nullptr);
    struct tm* t = localtime(&now);
    char time_stamp_name[50];
    strftime(time_stamp_name, sizeof(time_stamp_name), "../models/%y-%m-%d_%H-%M-%S.model", t);

    // saving model
    std::cout << "LOG: ml-train: saving model into ../models/latest.model and " << time_stamp_name << "\n";
    safe_xgboost(XGBoosterSaveModel(booster, "../models/latest.model"));
    safe_xgboost(XGBoosterSaveModel(booster, time_stamp_name));
}

// R is the ml reduced graph
// reverse_mapping is the mapping from old ids to new ids (rev[old] = new)
void ml_reducer::ml_reduce_old(graph_access& G, graph_access& R, std::vector<NodeID>& reverse_mapping) {
    ml_features feature_mat = ml_features(mis_config, G);
    feature_mat.initDMatrix();

    // predict
    bst_ulong out_len = 0;
    const float* _prediction = nullptr;
    safe_xgboost(XGBoosterPredict(booster, feature_mat.getDMatrix(), 0, 0, 0, &out_len, &_prediction));
    ASSERT_EQ(out_len, G.number_of_nodes());

    std::vector<float> prediction;
    prediction.assign(_prediction, _prediction+out_len);

    // sort nodes by their prediction
    std::vector<NodeID> high_candidates = std::vector<NodeID>(G.number_of_nodes(), 0);
    std::iota(high_candidates.begin(), high_candidates.end(), 0);
    std::sort(high_candidates.begin(), high_candidates.end(),
              [&prediction](const NodeID& n1, const NodeID& n2){ return prediction[n1] < prediction[n2]; }
    );

    std::vector<bool> exists(G.number_of_nodes(), true);
    // std::vector<NodeID> force_IS;

    // isolate high nodes, remove low nodes
    for(auto node : high_candidates) {
        if (exists[node]) {
            if (prediction[node] >= q) {
                // force_IS.push_back(node);
                // exists[node] = false;
                forall_out_edges(G, edge, node) {
                            NodeID neighbor = G.getEdgeTarget(node);
                            exists[neighbor] = false;
                        } endfor
            }
            if (prediction[node] < 1-q) {
                exists[node] = false;
            }
        }
    }

    // map force_IS through all reverse mappings, by iterating through the mappings in reverse and mapping the node_ids
    // save force_IS in the original graph
    // check if IS in original graph is still independent

    // calculate number of edges and create set of nodes for R
    EdgeID R_m = 0;
    std::vector<NodeID> R_nodes; // is mapping from R_ids to original ids
    R_nodes.reserve(G.number_of_nodes());
    forall_nodes(G, node) {
                if (exists[node]) {
                    forall_out_edges(G, edge, node) {
                                if (exists[node])
                                    ++R_m;
                            } endfor
                    R_nodes.push_back(node);
                }
            } endfor

    // make reverse mapping from original graph to R
    reverse_mapping.resize(G.number_of_nodes(), -1);   // if node does not exist anymore, it is mapped to -1
    for (int new_id = 0; new_id < R_nodes.size(); ++new_id) {
        reverse_mapping[R_nodes[new_id]] = new_id;
    }

    // construct R
    R.start_construction(R_nodes.size(), R_m);
    for (auto node : R_nodes) {
        NodeID new_node = R.new_node();
        R.setNodeWeight(new_node, G.getNodeWeight(node));

        forall_out_edges(G, edge, node) {
            auto new_target = reverse_mapping[G.getEdgeTarget(edge)];
            if (new_target != -1)
                R.new_edge(new_node, reverse_mapping[G.getEdgeTarget(edge)]);
        } endfor
    }
    R.finish_construction();
}

NodeWeight ml_reducer::ml_reduce(graph_access& G, graph_access& R, std::vector<NodeID>& reverse_mapping) {
    ml_features feature_mat = ml_features(mis_config, G);
    feature_mat.initDMatrix();

    // predict
    bst_ulong out_len = 0;
    const float* _prediction = nullptr;
    safe_xgboost(XGBoosterPredict(booster, feature_mat.getDMatrix(), 0, 0, 0, &out_len, &_prediction));
    ASSERT_EQ(out_len, G.number_of_nodes());

    std::vector<float> prediction;
    prediction.assign(_prediction, _prediction+out_len);

    // sort nodes by their prediction
    std::vector<NodeID> high_candidates = std::vector<NodeID>(G.number_of_nodes(), 0);
    std::iota(high_candidates.begin(), high_candidates.end(), 0);
    std::sort(high_candidates.begin(), high_candidates.end(),
              [&prediction](const NodeID& n1, const NodeID& n2){ return prediction[n1] < prediction[n2]; }
    );

    std::vector<bool> exists(G.number_of_nodes(), true);
    NodeWeight is_weight = 0;

    // isolate high nodes, remove low nodes
    for(auto node : high_candidates) {
        if (exists[node]) {
            if (prediction[node] >= q) {
                G.setPartitionIndex(node, 1);
                is_weight += G.getNodeWeight(node);
                exists[node] = false;
                forall_out_edges(G, edge, node) {
                    NodeID neighbor = G.getEdgeTarget(node);
                    exists[neighbor] = false;
                } endfor
            }
            if (prediction[node] < 1-q) {
                exists[node] = false;
            }
        }
    }


    // calculate number of edges and create set of nodes for R
    EdgeID R_m = 0;
    std::vector<NodeID> R_nodes; // is mapping from R_ids to original ids
    R_nodes.reserve(G.number_of_nodes());
    forall_nodes(G, node) {
        if (exists[node]) {
            forall_out_edges(G, edge, node) {
                if (exists[node])
                    ++R_m;
            } endfor
            R_nodes.push_back(node);
        }
    } endfor

    // make reverse mapping from original graph to R
    reverse_mapping.resize(G.number_of_nodes(), -1);   // if node does not exist anymore, it is mapped to -1
    for (int new_id = 0; new_id < R_nodes.size(); ++new_id) {
        reverse_mapping[R_nodes[new_id]] = new_id;
    }

    // construct R
    R.start_construction(R_nodes.size(), R_m);
    for (auto node : R_nodes) {
        NodeID new_node = R.new_node();
        R.setNodeWeight(new_node, G.getNodeWeight(node));

        forall_out_edges(G, edge, node) {
            auto new_target = reverse_mapping[G.getEdgeTarget(edge)];
            if (new_target != -1)
                R.new_edge(new_node, reverse_mapping[G.getEdgeTarget(edge)]);
        } endfor
    }
    R.finish_construction();

    return is_weight;
}

NodeWeight ml_reducer::iterative_reduce(graph_access& orignal_graph) {
    graph_access* G = &orignal_graph;
    graph_access R;

    NodeWeight current_weight = 0;
    std::vector<std::vector<NodeID>> mappings;
    std::vector<std::vector<NodeID>> independent_sets;

    // map force_IS through all reverse mappings, by iterating through the mappings in reverse and mapping the node_ids
    // save force_IS in the original graph
    // check if IS in original graph is still independent
    for (int iteration = 0; G->number_of_nodes() > 0; ++iteration) {
        std::vector<NodeID> reverse_mapping(G->number_of_nodes(), -1);

        if (iteration % 2 == 0) { // even -> kamis
            branch_and_reduce_algorithm kamis_reducer(*G, mis_config);
            kamis_reducer.reduce_graph();
            kamis_reducer.build_graph_access(R, reverse_mapping);
            current_weight += kamis_reducer.get_current_is_weight();

        } else { // odd -> ml
            current_weight += ml_reduce(*G, R, reverse_mapping);
        }
        // save IS nodes into vector, push onto stack
        mappings.push_back(reverse_mapping);

        std::vector<NodeID> force_IS;
        for (NodeID node = 0; node < G->number_of_nodes(); ++node) {
            if (G->getPartitionIndex(node) == 1)  // node is in IS
                force_IS.push_back(node);
        }
        independent_sets.push_back(force_IS);

        G = &R;
    }

    std::vector<NodeID> current_IS;

    for (int i = mappings.size(); i > 0; --i) {  // iterate backwards through the mappings
        auto mapping = mappings[i];  // TODO: these need to be from new to old ID's

        std::vector<NodeID> translated;
        std::copy(independent_sets[i].begin(), independent_sets[i].end(), std::back_inserter(translated));

        for (auto node : current_IS) {
            translated.push_back(mapping[node]);
        }

        current_IS = translated;
    }

    is_IS(*G);

    return 0;
}
