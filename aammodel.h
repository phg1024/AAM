#pragma once

#include "common.h"

class AAMModel {
public:
  AAMModel(){}
  AAMModel(const std::vector<QImage>& images, const std::vector<Eigen::MatrixX2d>& points);
  ~AAMModel(){}

  void BuildModel(const std::vector<int>& indices = std::vector<int>());

protected:
  void Preprocess();

private:
  // Input data
  std::vector<QImage> input_images;
  std::vector<Eigen::MatrixX2d> input_points;

  // Converted data
  std::vector<cv::Mat> images;
  Eigen::MatrixXd shapes;
};
