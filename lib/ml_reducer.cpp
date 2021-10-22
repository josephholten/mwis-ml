//
// Created by joseph on 10/16/21.
//

#include <numeric>
#include "ml_reducer.h"
#include "ml_features.h"

#include "safe_c_api.h"

ml_reducer::ml_reducer(MISConfig _mis_config, graph_access& _graph, const float _q) : G {_graph}, mis_config {std::move(_mis_config)}, q {_q} {

}

void ml_reducer::ml_reduce(graph_access& R, std::vector<NodeID>& reverse_mapping) { // R is the ml reduced graph
    // XGBoost predictor
    BoosterHandle predictor;
    safe_xgboost(XGBoosterCreate(nullptr, 0, &predictor));

    safe_xgboost(XGBoosterSetParam(predictor, "eta", "1"));

    // safe_xgboost(XGBoosterSetParam(predictor, "nthread", "16"));

    XGBoosterLoadModel(predictor, "models/standard.model");

    bst_ulong num_of_features;
    safe_xgboost(XGBoosterGetNumFeature(predictor, &num_of_features));
    // std::cout << num_of_features << "\n";

    // calculate features, storing in feature_mat
    DMatrixHandle feature_dmat;

    std::vector<float> feat_mat((size_t) G.number_of_nodes() * FEATURE_NUM, 0);
    features(mis_config, G, feat_mat.begin());
    safe_xgboost(XGDMatrixCreateFromMat(&feat_mat[0], (bst_ulong) G.number_of_nodes(), (bst_ulong) FEATURE_NUM, 0, &feature_dmat));

    // predict
    bst_ulong out_len = 0;
    const float* _prediction = nullptr;
    safe_xgboost(XGBoosterPredict(predictor, feature_dmat, 0, 0, 0, &out_len, &_prediction));
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
        NodeID node = R_nodes[new_id];
        reverse_mapping[node] = new_id;
    }

    // construct R
    R.start_construction(R_nodes.size(), R_m);
    for (auto node : R_nodes) {
        NodeID new_node = R.new_node();
        R.setNodeWeight(new_node, G.getNodeWeight(node));

        forall_out_edges(G, edge, node) {
                    R.new_edge(new_node, reverse_mapping[G.getEdgeTarget(edge)]);
                } endfor
    }
    R.finish_construction();
}

void ml_reducer::iterative_reduce() {
    graph_access original_graph;
    // if (mis_config.write_IS) {
        G.copy(original_graph);
    // }

    NodeWeight current_weight = 0;
    std::vector<std::vector<NodeID>> mappings;
    std::vector<std::vector<NodeID>> independent_sets;

    int iteration = 0;
    while (G.number_of_nodes() > 0) {
        graph_access R; // currently reduced graph
        std::vector<NodeID> reverse_mapping(G.number_of_nodes(), 0);
        if (iteration % 2 == 0) { // even -> kamis
            branch_and_reduce_algorithm kamis_reducer(G, mis_config);
            kamis_reducer.reduce_graph();
            kamis_reducer.build_graph_access(R, reverse_mapping);

            current_weight += kamis_reducer.get_current_is_weight();

        } else { // odd -> ml
            ml_reduce(R, reverse_mapping);
            // current_weight += G.getISWeight();
        }
        mappings.push_back(reverse_mapping);   // TODO: think need to save the "true" mapping (from new to old ID's)
        // save IS nodes into vector, push onto stack

        R.copy(G); // copy R into G?
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

    // if write:
    //     graph_io::writeVector(current_IS)
}
