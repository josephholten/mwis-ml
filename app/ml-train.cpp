//
// Created by joseph on 10/14/21.
//
#include "mis_config.h"

#include "ml/ml_reducer.h"

int main(int argn, char** argv) {
    MISConfig mis_config;

    // parse parameters

    ml_reducer reducer(mis_config, 0.95);
    reducer.train_model();
    reducer.save_model();
}
