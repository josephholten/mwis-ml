//
// Created by joseph on 9/28/21.
//

#include <numeric>
#include "tools/safe_c_api.h"

#include "mis_config.h"
#include "graph_io.h"
#include "graph_access.h"

#include "ml/ml_features_old.h"


int main(int argn, char** argv) {
    // mis_log::instance()->restart_total_timer();

    MISConfig mis_config;
    std::string graph_filepath("/home/joseph/uni/sem5/christian/mwis-ml/instances/adjnoun.graph");

    // Parse the command line parameters;
    /*
    int ret_code = parse_parameters(argn, argv, mis_config, graph_filepath);
    if (ret_code) {
        return 0;
    }
     */

    float q = 0.95;  // TODO: commandline parameter

    // mis_config.graph_filename = graph_filepath.substr(graph_filepath.find_last_of( '/' ) +1);
    // mis_log::instance()->set_config(mis_config);

    // Read the graph
    graph_access G;
    std::string comments;
    graph_io::readGraphWeighted(G, graph_filepath);

    // mis_log::instance()->set_graph(G);

    // KaMIS reducer
    // std::unique_ptr<branch_and_reduce_algorithm> reducer;

    // XGBoost predictor
    BoosterHandle predictor;
    safe_xgboost(XGBoosterCreate(NULL, 0, &predictor));

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

    graph_access R;  // ml-reduced graph
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
    std::vector<NodeID> reverse_mapping(G.number_of_nodes(), -1);   // if node does not exist anymore, it is mapped to -1
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

    // write reduced graph to file
    graph_io::writeGraphWeighted(R, mis_config.output_filename);

    /*
    graph_access original;
    // read graph

    graph_access G = original;  // current graph
    NodeWeight weight_offset = 0;

    typedef mapping sized_vector<NodeID>;
    std::vector<mapping> reverse_mappings;

    do {
        graph_access K;
        weight_offset += perform_reduction(reducer, G, K, mis_config);

        // ml reduce
        features(mis_config, K, feature_mat);
        safe_xgboost(XGDMatrixCreateFromMat(feature_mat, G.number_of_nodes(), FEATURE_NUM, 0, &dmat));

        // kamis reduce

        // keep track of weight offset, and graph hierarchy for reverse reductions
    } while (hierarchy.top()->number_of_nodes() > 0);
     */

    // reverse reduce
    // if write:

    //    write solution to file

    safe_xgboost(XGBoosterFree(predictor));
    safe_xgboost(XGBoosterFree(feature_dmat));


}