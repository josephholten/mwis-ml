//
// Created by joseph on 10/22/21.
//

#ifndef MWIS_ML_SAFE_C_API_H
#define MWIS_ML_SAFE_C_API_H

#include "xgboost/c_api.h"

#define safe_xgboost(call) {                                            \
int err = (call);                                                       \
if (err != 0) {                                                         \
    throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
                        ": error in " + #call  + ":" + XGBGetLastError()); \
    }                                                                    \
}

#endif //MWIS_ML_SAFE_C_API_H
