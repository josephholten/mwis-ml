//
// Created by joseph on 10/29/21.
//

#ifndef MWIS_ML_MLCONFIG_H
#define MWIS_ML_MLCONFIG_H


#include <mis_config.h>

struct BoosterConfig {

};

struct MLConfig {
    BoosterConfig booster_config;

    float q = 0.95;
    std::string path = "../models/standard.model";
    bool console_log = false;
    bool timer_log = false;

    MISConfig kamis_reduction_config;
    MISConfig ls_config;
};


#endif //MWIS_ML_MLCONFIG_H
