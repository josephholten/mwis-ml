//
// Created by joseph on 10/14/21.
//

#ifndef MWIS_ML_STAT_H
#define MWIS_ML_STAT_H

template<class T1, class T2>
double chi2(T1 obs, T2 exp) {
    if (exp == 0)
        return 0;
    double diff = (double) obs - (double) exp;
    return (diff * diff) / exp;
}

#endif //MWIS_ML_STAT_H
