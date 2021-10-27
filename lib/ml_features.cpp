//
// Created by joseph on 10/22/21.
//

#include "ml_features.h"

#include <sstream>
#include <fstream>

#include <numeric>
#include <algorithm>
#include <unordered_set>
#include <random>

#include "mis_config.h"
#include "configuration_mis.h"
#include "graph_access.h"
#include "wmis_interface/weighted_ls.h"
#include "tools/stat.h"
#include "tools/timer.h"


// TODO: extract algorithms

void features(MISConfig& mis_config, graph_access& G, std::vector<float>::iterator feat_mat) {
    timer t;

    // precalculation
    const NodeID number_of_nodes = G.number_of_nodes();
    const EdgeID number_of_edges = G.number_of_edges();

    NodeWeight total_weight = 0;
    forall_nodes(G, node) {
        total_weight += G.getNodeWeight(node);
    } endfor

    // greedy node coloring
    std::vector<int> node_coloring(G.number_of_nodes());
    std::vector<bool> available(G.number_of_nodes(), true);
    std::cout << "LOG: ml-features: starting greedy node coloring  ..." << std::flush;

    forall_nodes(G, node) {
        std::fill(available.begin(), available.end(), true);
        forall_out_edges(G, edge, node) {
            available[node_coloring[G.getEdgeTarget(edge)]] = false;
        } endfor
        node_coloring[node] = (int) (std::find_if(available.begin(), available.end(), [](bool x){ return x; }) - available.begin());
    } endfor
    int greedy_chromatic_number = *std::max_element(node_coloring.begin(), node_coloring.end()) + 1;
    std::vector<bool> used_colors(greedy_chromatic_number);
    std::cout << " done.\n";

    // local search
    std::vector<int> ls_signal(G.number_of_nodes(), 0);
    const int ls_rounds = 5;  // TODO: in config
    configuration_mis cfg;
    cfg.standard(mis_config);
    mis_config.console_log = false;
    mis_config.time_limit = 5.0;

    std::random_device rd;

    // TODO: log correctly
    for (int round = 0; round < ls_rounds; ++round) {
        mis_config.seed = (int) rd();
        std::cout << "LOG: ml-features: starting ls round " << round << " ... " << std::flush;
        t.restart();
        weighted_ls ls(mis_config, G);
        ls.run_ils();
        auto t_ils = t.elapsed();
        t.restart();
        forall_nodes(G, node) {
            ls_signal[node] += (int) G.getPartitionIndex(node);
        } endfor
        std::cout << "done"
                  << " (ils: " << t_ils << ", signal: " << t.elapsed() << "s)"
                  << ".\n";
    }

    // loop variables
    EdgeID local_edges = 0;
    std::unordered_set<NodeID> neighbors {};      // don't know how to do faster? maybe using bitset from boost?
    neighbors.reserve(G.getMaxDegree()+1);
    float avg_lcc = 0;
    float avg_wdeg = 0;

    std::cout << "LOG: ml-features: starting filling matrix ..." << std::flush;
    t.restart();
    forall_nodes(G, node){
        const NodeID row_start = node*FEATURE_NUM;

        // num of nodes/ edges, deg
        feat_mat[row_start + NODES] = (float) number_of_nodes;
        feat_mat[row_start + EDGES] = (float) number_of_edges;
        feat_mat[row_start + DEG] = (float) G.getNodeDegree(node);

        // lcc
        neighbors.clear();
        forall_out_edges(G, edge, node) {
            neighbors.insert(G.getEdgeTarget(edge));
        } endfor
        for (auto& neighbor : neighbors) {
            forall_out_edges(G, neighbor_edge, neighbor) {
                if (neighbors.find(G.getEdgeTarget(neighbor_edge)) != neighbors.end())
                    ++local_edges;
            } endfor
        }

        if (G.getNodeDegree(node) > 1)  // catch divide by zero
            feat_mat[row_start + LCC] = ((float) local_edges) / ((float) (G.getNodeDegree(node) * (G.getNodeDegree(node) - 1)));
        else
            feat_mat[row_start + LCC] = 0;
        avg_lcc += feat_mat[row_start + LCC];

        // local chromatic density
        forall_out_edges(G, edge, node) {
            used_colors[node_coloring[G.getEdgeTarget(edge)]] = true;
        } endfor
                feat_mat[row_start + CHROMATIC] = (float) std::accumulate(used_colors.begin(), used_colors.end(), 0) / (float) greedy_chromatic_number;

        // total weight
        feat_mat[row_start + T_WEIGHT] = (float) total_weight;

        // node weight
        feat_mat[row_start + NODE_W] = (float) G.getNodeWeight(node);

        // node weighted degree
        forall_out_edges(G, edge, node) {
            feat_mat[row_start + W_DEG] += (float) G.getNodeWeight(G.getEdgeTarget(edge));
        } endfor
        avg_wdeg += feat_mat[row_start + W_DEG];

        // local search
        feat_mat[row_start + LOCAL_SEARCH] += (float) ls_signal[node] / ls_rounds;

    } endfor

    avg_lcc /= (float) G.number_of_nodes();
    avg_wdeg /= (float) G.number_of_nodes();

    // statistical features
    float avg_deg = (2 * (float) G.number_of_edges()) / ((float) G.number_of_nodes());

    forall_nodes(G,node) {
        const NodeID row_start = node*FEATURE_NUM;
        feat_mat[row_start + CHI2_DEG] = (float) chi2(G.getNodeDegree(node), avg_deg);

        float avg_chi2_deg = 0;
        forall_out_edges(G, edge, node) {
            avg_chi2_deg += feat_mat[G.getEdgeTarget(edge) * FEATURE_NUM + AVG_CHI2_DEG];
        } endfor

        if(G.getNodeDegree(node) > 0)    // catch divide by zero
            feat_mat[row_start + AVG_CHI2_DEG] = avg_chi2_deg / (float) G.getNodeDegree(node);
        else
            feat_mat[row_start + AVG_CHI2_DEG] = 0;

        feat_mat[row_start + CHI2_LCC] = (float) chi2(feat_mat[row_start + LCC], avg_lcc);

        feat_mat[row_start + CHI2_W_DEG] = (float) chi2(feat_mat[row_start + W_DEG], avg_wdeg);
    } endfor
    std::cout << "done"
              << " (" << t.elapsed() <<  "s)"
              << ".\n";
}
