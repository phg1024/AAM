#include "aammodel.h"

using namespace std;

AAMModel::AAMModel(const std::vector<QImage>& images, const std::vector<Eigen::Matrix2d>& points)
  : input_images(images), input_points(points) {
  Preprocess();
}

void AAMModel::Preprocess() {
  // Convert input images to opencv Mat

  // Collect all input shapes
}

void AAMModel::BuildModel(const vector<int>& indices) {
  // Compute mean shape

  // Compute mean texture

  // Construct shape and texture model
}
