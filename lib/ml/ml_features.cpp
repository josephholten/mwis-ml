//
// Created by joseph on 10/27/21.
//

#include "ml_features.h"

#include <random>
#include <unordered_set>
#include <omp.h>

#include "mis_config.h"
#include "configuration_mis.h"
#include "timer.h"
#include "graph_io.h"
#include "wmis_interface/weighted_ls.h"
#include "tools/stat.h"

// constructors
// for training (many graphs with labels)
ml_features::ml_features(MISConfig config)
    : feature_matrix(FEATURE_NUM), has_labels {true}, mis_config {std::move(config)}
{}

// for reducing (single graph without labels)
ml_features::ml_features(MISConfig config, graph_access &G)
    : feature_matrix(FEATURE_NUM), has_labels {false}, mis_config {std::move(config)}
{
    reserveNodes(G.number_of_nodes());
    fillGraph(G);
}

ml_features::~ml_features() {
    XGDMatrixFree(dmat);
}

// getters
int ml_features::getNumberOfFeatures() {
    return FEATURE_NUM;
}

void ml_features::fromPaths(const std::vector<std::string> &graph_paths, const std::vector<std::string>& label_paths) {
    assert((graph_paths.size() == label_paths.size()) && "Error: provide same number of graph and label paths\n");

    // calculate the node offsets of each graph
    // produces a mapping of index in the list of paths (provided by the user) and the start position in the feature_matrix and label_data
    std::vector<NodeID> offsets(graph_paths.size()+1,0);
    for (int i = 0; i < graph_paths.size(); i++) {
        offsets[i+1] = offsets[i] + graph_io::readNumberOfNodes(graph_paths[i]);
    }

    // last offset is total number of nodes
    reserveNodes(offsets[graph_paths.size()]);

    //#pragma omp parallel for default(none) shared(graph_paths, label_paths, offsets, std::cout)
    for (int i = 0; i < graph_paths.size(); ++i) {
        std::cout << "Thread " << omp_get_thread_num() << std::endl;
        graph_access G;
        graph_io::readGraphWeighted(G, graph_paths[i]);
        std::vector<float> labels(G.number_of_nodes(), 0);
        graph_io::readVector<float>(labels.begin(), offsets[i+1]-offsets[i], label_paths[i]);

        fillGraph(G, labels, offsets[i]);
    }
}

// init
void ml_features::reserveNodes(NodeID n) {
    feature_matrix.addRows(n);
    label_data.resize(label_data.size() + n);
}

void ml_features::fillGraph(graph_access& G) {
    if (has_labels) {
        std::cerr << "Error: feature matrix was constructed for labels, so provide the labels for the graph.\n";
        exit(1);
    }
    features(G);
}

void ml_features::fillGraph(graph_access& G, std::vector<float>& labels, NodeID offset) {
    if (!has_labels) {
        std::cerr << "Error: feature matrix was constructed not for labels, so the labels will be discarded.\n";
    }
    features(G);
    std::copy(labels.begin(), labels.end(), label_data.begin() + offset);
}



template<ml_features::feature f>
float& ml_features::getFeature(NodeID node) {
    return feature_matrix[node + current_size][f];
}

void ml_features::features(graph_access& G) {
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
    std::cout << "LOG: ml-features: starting greedy node coloring  ... " << std::flush;

    forall_nodes(G, node) {
        std::fill(available.begin(), available.end(), true);
        forall_out_edges(G, edge, node) {
            available[node_coloring[G.getEdgeTarget(edge)]] = false;
        } endfor
        node_coloring[node] = (int) (std::find_if(available.begin(), available.end(), [](bool x){ return x; }) - available.begin());
    } endfor
    int greedy_chromatic_number = *std::max_element(node_coloring.begin(), node_coloring.end()) + 1;
    std::vector<bool> used_colors(greedy_chromatic_number);
    std::cout << "done.\n";

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

    // loop variales
    EdgeID local_edges = 0;
    std::unordered_set<NodeID> neighbors {};      // don't know how to do faster? maybe using bitset from boost?
    neighbors.reserve(G.getMaxDegree()+1);
    float avg_lcc = 0;
    float avg_wdeg = 0;

    std::cout << "LOG: ml-features: starting filling matrix ..." << std::flush;
    t.restart();
    forall_nodes(G, node){
        // num of nodes/ edges, deg
        getFeature<NODES>(node) = (float) G.number_of_nodes();
        getFeature<EDGES>(node) = (float) G.number_of_edges();
        getFeature<DEG>(node) = (float) G.getNodeDegree(node);

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
            getFeature<LCC>(node) = ((float) local_edges) / ((float) (G.getNodeDegree(node) * (G.getNodeDegree(node) - 1)));
        else
            getFeature<LCC>(node) = 0;
        avg_lcc += getFeature<LCC>(node);

        // local chromatic density
        forall_out_edges(G, edge, node) {
            used_colors[node_coloring[G.getEdgeTarget(edge)]] = true;
        } endfor
        getFeature<CHROMATIC>(node) = (float) std::accumulate(used_colors.begin(), used_colors.end(), 0) / (float) greedy_chromatic_number;

        // total weight
        getFeature<T_WEIGHT>(node) = (float) total_weight;

        // node weight
        getFeature<NODE_W>(node) = (float) G.getNodeWeight(node);

        // node weighted degree
        forall_out_edges(G, edge, node) {
            getFeature<W_DEG>(node) += (float) G.getNodeWeight(G.getEdgeTarget(edge));
        } endfor
        avg_wdeg = getFeature<W_DEG>(node);

        // local search
        getFeature<LOCAL_SEARCH>(node) += (float) ls_signal[node] / ls_rounds;

    } endfor

    avg_lcc /= (float) G.number_of_nodes();
    avg_wdeg /= (float) G.number_of_nodes();

    // statistical features
    float avg_deg = (2 * (float) G.number_of_edges()) / ((float) G.number_of_nodes());

    forall_nodes(G,node) {
        getFeature<CHI2_DEG>(node) = (float) chi2(G.getNodeDegree(node), avg_deg);

        float avg_chi2_deg = 0;
        forall_out_edges(G, edge, node) {
            avg_chi2_deg += getFeature<AVG_CHI2_DEG>(node);
        } endfor

        if(G.getNodeDegree(node) > 0)    // catch divide by zero
            getFeature<AVG_CHI2_DEG>(node) = avg_chi2_deg / (float) G.getNodeDegree(node);
        else
            getFeature<AVG_CHI2_DEG>(node) = 0;

        getFeature<CHI2_LCC>(node) = (float) chi2(getFeature<LCC>(node), avg_lcc);

        getFeature<CHI2_W_DEG>(node) = (float) chi2(getFeature<W_DEG>(node), avg_wdeg);
    } endfor

    std::cout << "done"
              << " (" << t.elapsed() <<  "s)"
              << ".\n";

    current_size += G.number_of_nodes();
}

void ml_features::initDMatrix() {
    XGDMatrixCreateFromMat(feature_matrix.c_arr(), feature_matrix.getRows(), feature_matrix.getCols(), 0, &dmat);
    XGDMatrixSetFloatInfo(dmat, "label", &label_data[0], label_data.size());
}

DMatrixHandle ml_features::getDMatrix() {
    return dmat;
}




