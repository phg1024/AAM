#pragma once

#include "common.h"

std::vector<std::string> ReadFileByLine(const std::string &filename);
std::vector<std::pair<std::string, std::string>> ParseSettingsFile(const std::string& settings_filename);
std::pair<QImage, Eigen::MatrixX2d> LoadImagePointsPair(const std::string& image_filename, const std::string& points_filename);
