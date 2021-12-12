//
// Created by joseph on 10/14/21.
//
#include "mis_config.h"

#include "ml/ml_reducer.h"

int main(int argn, char** argv) {
    MISConfig mis_config;
    mis_config.q = 0.95;

    // parse parameters

    ml_reducer reducer(mis_config);
    reducer.train_model();
    reducer.save_model();
}
