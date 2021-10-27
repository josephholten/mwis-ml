//
// Created by joseph on 10/22/21.
//

#ifndef MWIS_ML_IO_WRAPPER_H
#define MWIS_ML_IO_WRAPPER_H

void features_from_paths(MISConfig& mis_config, const std::vector<std::string>& paths, std::vector<float>& feat_mat);
void labels_from_paths(const std::vector<std::string>& paths, typename std::vector<float>::iterator label_vec);
std::vector<std::string> split_file_by_lines(const std::string& path);


#endif //MWIS_ML_IO_WRAPPER_H
