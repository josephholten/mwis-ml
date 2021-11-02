//
// Created by joseph on 10/27/21.
//

#ifndef MWIS_ML_ML_FEATURES_H
#define MWIS_ML_ML_FEATURES_H

#include <vector>
#include <mis_config.h>

#include "graph_access.h"
#include "tools/safe_c_api.h"

class matrix {
public:
    explicit matrix(size_t _cols) : cols {_cols} {};

    void addRows(size_t _rows) {
        rows += _rows;
        data.resize(data.size() + rows*cols);
    }

    std::vector<float>::iterator operator[](size_t idx) { return data.begin() + idx*cols; };

    size_t getRows() { return rows; };
    size_t getCols() { return cols; };

    float* c_arr() { return &data[0]; };

private:
    size_t cols {};
    size_t rows {};

    std::vector<float> data;
};

class ml_features {
public:
    explicit ml_features(MISConfig config, graph_access& G);   // for single graph
    explicit ml_features(MISConfig config);    // for multiple graphs
    ~ml_features();

    [[nodiscard]] static int getNumberOfFeatures() ;
    size_t getRows() { return feature_matrix.getRows(); };
    size_t getCols() { return feature_matrix.getCols(); };

    void fromPaths(const std::vector<std::string> &graphs, const std::vector<std::string> &labels);

    void reserveNodes(NodeID n);   // for reserving memory
    void fillGraph(graph_access& G);
    void fillGraph(graph_access& G, std::vector<float>& labels);

    void initDMatrix();
    DMatrixHandle getDMatrix();


private:
    static const int FEATURE_NUM = 13;
    enum feature : int { NODES=0, EDGES=1, DEG=2, CHI2_DEG=3, AVG_CHI2_DEG=4, LCC=5, CHI2_LCC=6, CHROMATIC=7, T_WEIGHT=8, NODE_W=9, W_DEG=10, CHI2_W_DEG=11, LOCAL_SEARCH=12 };

    MISConfig mis_config;
    size_t current_size {0};
    matrix feature_matrix;
    bool has_labels;
    std::vector<float> label_data;

    // TODO: private DMatrixHandle variable with custom destructor freeing the handle

    template<feature f>
    float& getFeature(NodeID node);

    void features(graph_access& G);

    DMatrixHandle dmat;

};

#endif //MWIS_ML_ML_FEATURES_H