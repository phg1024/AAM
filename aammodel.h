#pragma once

#include "common.h"

class AAMModel {
public:
  AAMModel(){}
  AAMModel(const std::vector<QImage>& images, const std::vector<cv::Mat>& points);
  ~AAMModel(){}

  void BuildModel(std::vector<int> indices = std::vector<int>());

protected:
  void Preprocess();

  cv::Mat ComputeMeanShape(const cv::Mat& shapes);
  cv::Mat AlignShape(const cv::Mat& from_shape,
                     const cv::Mat& to_shape);
  cv::Mat ScaleShape(const cv::Mat& shape, double size);

  cv::Mat ComputeMeanTexture(const std::vector<cv::Mat>& images,
                             const cv::Mat& shapes,
                             const cv::Mat& meanshape);

private:
  // Input data
  std::vector<QImage> input_images;
  std::vector<cv::Mat> input_points;

  // Converted data
  std::vector<cv::Mat> images, warped_images;
  cv::Mat shapes;
  cv::Mat textures, normalized_textures;

  std::vector<cv::Vec3i> triangles;

  // Affine transformation from each triangle to the meanshape in each image
  std::vector<std::vector<cv::Mat>> tforms, tforms_inv;

  cv::Mat pixel_map;
  std::vector<int> pixel_counts;
  std::vector<std::vector<cv::Vec2i>> pixel_coords;
  std::vector<cv::Mat> pixel_mats;

  cv::Mat meanshape, meantexture;
};
