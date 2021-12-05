//
// Created by joseph on 9/28/21.
//

#include <ml/ml_reducer.h>

#include "mis_config.h"
#include "graph_io.h"
#include "graph_access.h"
#include "parse_parameters.h"

int main(int argn, char** argv) {
    MISConfig mis_config;
    std::string graph_filepath;

    // Parse the command line parameters;
    int ret_code = parse_parameters(argn, argv, mis_config, graph_filepath);
    if (ret_code) {
        return 0;
    }

    float q = 0.95;  // TODO: commandline parameter

    mis_config.graph_filename = graph_filepath.substr(graph_filepath.find_last_of('/') + 1);

    // Read the graph
    graph_access G;
    graph_io::readGraphWeighted(G, graph_filepath);

    ml_reducer reducer = ml_reducer(mis_config, q);
    graph_access R;
    std::vector<NodeID> reverse_mapping;
    reducer.ml_reduce(G, R, reverse_mapping);

    graph_io::writeGraphWeighted(R, mis_config.output_filename);
}
