int example(int argn, char **argv) {
    mis_log::instance()->restart_total_timer();
    //mis_log::instance()->print_title();

    MISConfig mis_config;
    std::string graph_filepath;

    // Parse the command line parameters;
    int ret_code = parse_parameters(argn, argv, mis_config, graph_filepath);
    if (ret_code) {
        return 0;
    }

    mis_config.graph_filename = graph_filepath.substr(graph_filepath.find_last_of( '/' ) +1);
    mis_log::instance()->set_config(mis_config);

    // Read the graph
    graph_access G;
    std::string comments;
    graph_io::readGraphWeighted(G, graph_filepath, comments);

    mis_log::instance()->set_graph(G);

    NodeWeight weight_offset = 0;
    std::unique_ptr<branch_and_reduce_algorithm> reducer;

    //if (mis_config.write_graph) {
    //// just reduce the graph and write it into a file
    //graph_access rG;

    //auto start = std::chrono::system_clock::now();
    //weight_offset = perform_reduction(reducer, G, rG, mis_config);
    //auto end = std::chrono::system_clock::now();

    //std::chrono::duration<float> reduction_time = end - start;

    //std::ofstream output_reduced(mis_config.output_filename);

    //output_reduced << "%reduction_time " << reduction_time.count() << "\n";
    //output_reduced << "%reduction_offset " << weight_offset << "\n";

    //graph_io::writeGraphNodeWeighted(rG, output_reduced);
    //return 0;
    //}

    //std::cout << "%nodes " << G.number_of_nodes() << std::endl;

    // if (mis_config.perform_reductions) {
    // recude graph and run local search
    graph_access rG;

    auto start = std::chrono::system_clock::now();
    weight_offset = perform_reduction(reducer, G, rG, mis_config);
    auto end = std::chrono::system_clock::now();

    std::chrono::duration<float> reduction_time = end - start;

    //std::cout << "%reduction_nodes " << rG.number_of_nodes() << "\n";
    //std::cout << "%reduction_time " << reduction_time.count() << "\n";
    //std::cout << "%reduction_offset " << weight_offset << std::endl;

    if (rG.number_of_nodes() != 0) {
        perform_ils(mis_config, rG, weight_offset);
    } else {
        std::cout << "MIS_weight " << weight_offset << std::endl;
    }

    reducer->reverse_reduction(G, rG, reverse_mapping);

    if (!is_IS(G)) {
        std::cerr << "ERROR: graph after inverse reduction is not independent" << std::endl;
        exit(1);
    } else {
        NodeWeight is_weight = 0;

        forall_nodes(G, node) {
            if (G.getPartitionIndex(node) == 1) {
                is_weight += G.getNodeWeight(node);
            }
        } endfor

        std::cout << "MIS_weight_check " << is_weight << std::endl;
    }
    // } else {
    // run local search whithout reductions
    weight_offset = extractReductionOffset(comments);

    std::cout << comments;
    perform_ils(mis_config, G, weight_offset);
    // }

    if (mis_config.write_graph) graph_io::writeIndependentSet(G, mis_config.output_filename);

    return 0;
}
