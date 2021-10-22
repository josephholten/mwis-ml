/**
 * reduction_evomis.cpp
 * Purpose: Main program for the evolutionary algorithm.
 *
 *****************************************************************************/

#include <iostream>
#include <random>

#include "ils/ils.h"
#include "graph_access.h"
#include "mis_config.h"
#include "branch_and_reduce_algorithm.h"


void initial_is(graph_access& G) {
	std::vector<NodeID> nodes(G.number_of_nodes());
	for (size_t i = 0; i < nodes.size(); i++) {
		nodes[i] = i;
	}

	// sort in descending order by node weights
	std::sort(nodes.begin(), nodes.end(), [&G](NodeID lhs, NodeID rhs) {
		return G.getNodeWeight(lhs) > G.getNodeWeight(rhs);
	});

	for (NodeID n : nodes) {
		bool free_node = true;

		forall_out_edges(G, edge, n) {
			NodeID neighbor = G.getEdgeTarget(edge);
			if (G.getPartitionIndex(neighbor) == 1) {
				free_node = false;
				break;
			}
		} endfor

		if (free_node) G.setPartitionIndex(n, 1);
	}
}

bool is_IS(graph_access& G) {
	forall_nodes(G, node) {
		if (G.getPartitionIndex(node) == 1) {
			forall_out_edges(G, edge, node) {
				NodeID neighbor = G.getEdgeTarget(edge);
				if (G.getPartitionIndex(neighbor) == 1) {
					return false;
				}
			} endfor
		}
	} endfor

	return true;
}

std::vector<NodeID> reverse_mapping;
NodeWeight perform_reduction(std::unique_ptr<branch_and_reduce_algorithm>& reducer, graph_access& G, graph_access& rG, const MISConfig& config) {
	reducer = std::unique_ptr<branch_and_reduce_algorithm>(new branch_and_reduce_algorithm(G, config));
	reducer->reduce_graph();

	// Retrieve reduced graph
	reverse_mapping = std::vector<NodeID>(G.number_of_nodes(), 0);
	reducer->build_graph_access(rG, reverse_mapping);

	if (!is_IS(rG)) {
		std::cerr << "ERROR: reduced graph is not independent" << std::endl;
		exit(1);
	}

	NodeWeight is_weight = reducer->get_current_is_weight();

	return is_weight;
}

void perform_ils(const MISConfig& mis_config, graph_access& G, NodeWeight weight_offset) {
	ils local(mis_config);
	initial_is(G);
	local.perform_ils(G, 1000000, weight_offset);

	if (!is_IS(G)) {
		std::cerr << "ERROR: graph after ILS is not independent" << std::endl;
		exit(1);
	}

	NodeWeight is_weight = weight_offset;

	forall_nodes(G, node) {
		if (G.getPartitionIndex(node) == 1) {
			is_weight += G.getNodeWeight(node);
		}
	} endfor

	// std::cout << "MIS_weight " << is_weight << std::endl;
}