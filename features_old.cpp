//
// Created by joseph on 14.06.21.
//

#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include <iterator>
#include <graph_access.h>
#include <graph_io.h>

#include <numeric>
#include <algorithm>
#include <unordered_set>

const int FEATURE_NUM = 13;
enum kw : int { NODES=0, EDGES=1, DEG=2, CHI2_DEG=3, AVG_CHI2_DEG=4, LCC=5, CHI2_LCC=6, CHROMATIC=7, T_WEIGHT=8, NODE_W=9, W_DEG=10, CHI2_W_DEG=11, LOCAL_SEARCH=12 };

class FeatureCalculator {
private:
    class feature_mat {
    public:
        typedef std::vector<double> feature_vec_t;

    private:
        NodeID m_nodes;
        std::vector<feature_vec_t> m_arr;
        std::vector<feature_vec_t> m_arr_T;

        std::vector<bool> m_filled;

        void transpose() {

            for (int i = 0; i < FEATURE_NUM; ++i) {
                for (NodeID node = 0; node < m_nodes; ++node) {
                    m_arr[node][i] = m_arr_T[i][node];
                }
            }
        }

    public:
        explicit feature_mat(NodeID nodes) :
                m_nodes(nodes),
                m_arr(nodes, feature_vec_t(FEATURE_NUM, 0.0)),
                m_arr_T(FEATURE_NUM, feature_vec_t(nodes, 0.0)),
                m_filled(FEATURE_NUM)
        { }

        ~feature_mat() = default;

        feature_vec_t& get_feature_col(kw name) {
            return m_arr_T.at(name);
        }

        template<class T>
        void fill_col(kw name, T value) {
            auto& col = get_feature_col(name);
            std::fill(col.begin(), col.end(), (double) value);
            m_filled[name] = true;
        }

        void filled_col(kw name) {
            m_filled[name] = true;
        }

        std::vector<feature_vec_t>& get_mat() {
            if (std::any_of(m_filled.begin(), m_filled.end(), [](bool x){ return !x;} ))
                throw std::out_of_range("matrix not ready, finalize computation first!");
            transpose();
            return m_arr;
        }
    } m_feature_mat;

    graph_access& m_G;

    template<class T>
    double average(std::vector<T>& vec) {
        if (vec.empty())
            return 0;
        auto sum = std::accumulate(vec.begin(), vec.end(), 0.0);
        return (double) std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();
    }

    template<class T1, class T2>
    double chi2(T1 obs, T2 exp) {
        if (exp == 0)
            return 0;
        double diff = obs - exp;
        return (diff * diff) / exp;
    }

    template<class T>
    void chi2(std::vector<T>& obs, std::vector<double>& res) {
        auto avg = average(obs);
        if (obs.size() != res.size())
            throw std::invalid_argument("sizes of obs and res don't match!");
        for (int i = 0; i < obs.size(); ++i) {
            res[i] = chi2(obs[i], avg);
        }
    }

    void local_clustering_coefficient(feature_mat::feature_vec_t& lcc) {
        forall_nodes(m_G, node) {
            EdgeID local_edges = 0;
            std::unordered_set<NodeID> neighbors {};      // don't know how to do faster? maybe using bitset from boost?
            neighbors.reserve(m_G.getNodeDegree(node)+1);
            forall_out_edges(m_G, edge, node) {
                neighbors.insert(m_G.getEdgeTarget(edge));
            } endfor
            for (auto& neighbor : neighbors) {
                forall_out_edges(m_G, neighbor_edge, neighbor) {
                    if (neighbors.find(m_G.getEdgeTarget(neighbor_edge)) != neighbors.end())
                        ++local_edges;
                } endfor
            }

            if (m_G.getNodeDegree(node) > 1)
                lcc[node] = ((double) local_edges) / (m_G.getNodeDegree(node) * (m_G.getNodeDegree(node) - 1));
            else
                lcc[node] = 0;
        } endfor
    }

    void greedy_coloring(std::vector<int>& coloring) {
        std::vector<bool> available(m_G.number_of_nodes(), true);
        forall_nodes(m_G, node) {
            std::fill(available.begin(), available.end(), true);
            forall_out_edges(m_G, edge, node) {
                available[coloring[m_G.getEdgeTarget(edge)]] = false;
            } endfor
            coloring[node] = (int) (std::find_if(available.begin(), available.end(), [](bool x){ return x; }) - available.begin());
        } endfor
    }


/*
        // (OLD) filling the feature array
        forall_nodes(G, node) {
            // F1: number of nodes
            features[node][0] = (double) number_of_nodes;
            // F2: number of edges
            features[node][1] = (double) number_of_edges;
            // F3: degree
            features[node][2] = (double) deg[node];
            // F4: local clustering coefficient
            features[node][3] = (double) lcc[node];
            // F5: skipped (eigencentrality)
            // stat. features
            // F6: chi2 of deg
            features[node][4] = chi2_deg[node];
            // F7: average chi2 of deg
            double sum = 0;
            forall_out_edges(G, e, node) {
                sum += chi2_deg[G.getEdgeTarget(e)];
            } endfor
            features[node][5] = sum / G.getNodeDegree(node);
            // F8: chi2 of lcc
            features[node][6] = chi2_lcc[node];
            // F9: average of chi2 of lcc
            sum = 0;
            forall_out_edges(G, e, node) {
                sum += chi2_lcc[G.getEdgeTarget(e)];
            } endfor
            features[node][7] = sum / G.getNodeDegree(node);
            // F10: local chromatic density estimate
            features[node][8] = local_chromatic_density_estimate[node];
            // F11: total node weight of graph
            features[node][9] = total_weight;
            // F12: node weight
            features[node][10] = G.getNodeWeight(node);
            // F13: node weighted degree (sum of node weights of neighbors)
            features[node][11] = node_weighted_degree[node];
            // F14: chi2 of node weighted degree
            features[node][12] = chi2_nwd[node];
        } endfor
         */


public:
    explicit FeatureCalculator(graph_access& G) : m_G(G), m_feature_mat(G.number_of_nodes()) {};
    ~FeatureCalculator() = default;

    void calc_features(const std::string& path, const std::string& temp_folder, int ls_time_limit) {
        std::string name(path.substr(path.find_last_of('/') + 1));

        // nodes
        m_feature_mat.fill_col(NODES, m_G.number_of_nodes());

        // edges
        m_feature_mat.fill_col(EDGES, m_G.number_of_edges());

        // deg
        auto& degree = m_feature_mat.get_feature_col(DEG);
        forall_nodes(m_G, node) {
            degree[node] = m_G.getNodeDegree(node);
        } endfor
        m_feature_mat.filled_col(DEG);
        chi2(degree, m_feature_mat.get_feature_col(CHI2_DEG));
        m_feature_mat.filled_col(CHI2_DEG);

        auto& avg_chi2_deg_col = m_feature_mat.get_feature_col(AVG_CHI2_DEG);
        forall_nodes(m_G, node) {
            double sum = 0;
            forall_out_edges(m_G, edge, node) {
                sum += m_feature_mat.get_feature_col(CHI2_DEG)[m_G.getEdgeTarget(edge)];
            } endfor
            avg_chi2_deg_col[node] = m_G.getNodeDegree(node) == 0 ? 0 : sum / m_G.getNodeDegree(node);
        } endfor
        m_feature_mat.filled_col(AVG_CHI2_DEG);

        // lcc
        local_clustering_coefficient(m_feature_mat.get_feature_col(LCC));
        m_feature_mat.filled_col(LCC);
        chi2(m_feature_mat.get_feature_col(LCC), m_feature_mat.get_feature_col(CHI2_LCC));
        m_feature_mat.filled_col(CHI2_LCC);

        // coloring
        std::vector<int> node_coloring(m_G.number_of_nodes());
        greedy_coloring(node_coloring);
        int greedy_chromatic_number = *std::max_element(node_coloring.begin(), node_coloring.end()) + 1;

        auto& local_chromatic_density_estimate = m_feature_mat.get_feature_col(CHROMATIC);
        forall_nodes(m_G, node) {
            std::vector<bool> used_colors(greedy_chromatic_number);
            forall_out_edges(m_G, edge, node) {
                used_colors[node_coloring[m_G.getEdgeTarget(edge)]] = true;
            } endfor
            local_chromatic_density_estimate[node] = (double) std::accumulate(used_colors.begin(), used_colors.end(), 0) / greedy_chromatic_number;
        } endfor
        m_feature_mat.filled_col(CHROMATIC);

        // total node weight
        NodeWeight total_weight = 0;
        forall_nodes(m_G, node) {
            total_weight += m_G.getNodeWeight(node);
        } endfor
        m_feature_mat.fill_col(T_WEIGHT, total_weight);

        auto& node_w = m_feature_mat.get_feature_col(NODE_W);
        forall_nodes(m_G, node) {
            node_w[node] = m_G.getNodeWeight(node);
        } endfor
        m_feature_mat.filled_col(NODE_W);

        // node weighted degree
        auto& node_weighted_degree = m_feature_mat.get_feature_col(W_DEG);
        forall_nodes(m_G, node) {
            forall_out_edges(m_G, edge, node) {
                node_weighted_degree[node] += m_G.getNodeWeight(m_G.getEdgeTarget(edge));
            } endfor
        } endfor
        m_feature_mat.filled_col(W_DEG);
        chi2(node_weighted_degree, m_feature_mat.get_feature_col(CHI2_W_DEG));
        m_feature_mat.filled_col(CHI2_W_DEG);


        // local search
        int ls_rounds = 5;
        auto& ls_col = m_feature_mat.get_feature_col(LOCAL_SEARCH);

        for (int round = 1; round <= ls_rounds; ++round) {
            std::stringstream ss;
            std::string ls_name;
            ss << temp_folder << "/" << name << ".w_ls" << ls_time_limit;
            ls_name = ss.str();
            ss.str("");
            ss << "deploy/weighted_local_search " << path
               << " --out="                       << ls_name
               << " --seed="                      << rand() % 100000
               << " --time_limit="                << ls_time_limit
               << " > /dev/null";  // > /dev/null to supress output

            std::system(ss.str().c_str());
            std::ifstream w_ls_file(ls_name);
            // read signal from cin and add the signal to ls_col
            std::transform(std::istream_iterator<int>(w_ls_file), std::istream_iterator<int>(),
                           ls_col.begin(), ls_col.begin(), std::plus<double>());
        }
        // divide the sum of signals by the number of signals
        std::transform(ls_col.begin(), ls_col.end(), ls_col.begin(), [ls_rounds](double signal_sum){ return signal_sum/ls_rounds; });
        m_feature_mat.filled_col(LOCAL_SEARCH);
    }

    void write_features(const std::string& path) {
        const auto& features = m_feature_mat.get_mat();
        std::ofstream file(path);
        for (const auto& feature_vec : features) {
            std::stringstream ss;
            for (auto value : feature_vec)
                ss << value << " ";
            ss << "\n";
            file << ss.str();
        }
    }
};


int main(int argn, char **argv) {
    // Parse the command line parameters;
    if (argn != 4) {
        std::cout << "please specify the graph path and the feature directory and the time limit for the local search" << "\n";
        return 1;
    }

    std::string graph_filepath(argv[1]);
    std::string graph_name(graph_filepath.substr(graph_filepath.find_last_of('/') + 1));
    std::string feature_directory(argv[2]);
    int ls_time_limit = std::stoi(argv[3]);

    graph_access G;
    graph_io::readGraphWeighted(G, graph_filepath);

    FeatureCalculator calc(G);
    calc.calc_features(graph_filepath, feature_directory, ls_time_limit);
    calc.write_features(feature_directory + "/" + graph_name + ".feat");
}
