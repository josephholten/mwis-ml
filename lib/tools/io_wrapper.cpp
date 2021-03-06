//
// Created by joseph on 10/22/21.
//

#include "mis_config.h"
#include "graph_access.h"
#include "graph_io.h"

/*
void features_from_paths(MISConfig& mis_config, const std::vector<std::string>& paths, std::vector<float>& feat_mat) {
    NodeID total_nodes = 0;
    for (const auto& path : paths) {
        total_nodes += graph_io::readNumberOfNodes(path);
    }
    // TODO: possibly make external memory an option when training on very large sets of graphs
    feat_mat.resize(total_nodes*FEATURE_NUM, 0);
    NodeID current_end = 0;
    for (const auto& path : paths) {
        graph_access G;
        int err = graph_io::readGraphWeighted(G, path);
        if (err)
            exit(1);
        std::cout << "LOG: ml-features: calculating features for " << path << "\n";
        features(mis_config, G, feat_mat.begin() + current_end);
        current_end += G.number_of_nodes() * FEATURE_NUM;
    }
}
 */

// void labels_from_paths(const std::vector<std::string>& paths, typename std::vector<float>::iterator label_vec) {
//     // assume user sizes label_vec accordingly
//     size_t current_end = 0;
//     for (const auto& path : paths) {
//         std::ifstream ifstream(path);
//         if (!ifstream) {
//             std::cerr << "Could not open file " << path << "\n";
//             exit(1);
//         }
//         current_end += graph_io::readVector<float>(label_vec + current_end, path);
//     }
// }

std::vector<std::string> split_file_by_lines(const std::string& path) {
    std::vector<std::string> lines;
    std::ifstream ifstream(path);
    if (!ifstream) {
        std::cerr << "Could not open file " << path << "\n";
        exit(1);
    }
    std::string line;
    while (std::getline(ifstream, line)) {
        if (line[0] == '#')
            continue;
        lines.push_back(line);
    }
    return lines;
}
