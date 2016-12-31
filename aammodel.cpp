#include "aammodel.h"
#include "utils.h"

using namespace std;

AAMModel::AAMModel(const std::vector<QImage>& images, const std::vector<Eigen::MatrixX2d>& points)
  : input_images(images), input_points(points) {
  Preprocess();
}

void AAMModel::Preprocess() {
  boost::timer::auto_cpu_timer t("Preprocessing finished in %w seconds.\n");

  const int nimages = input_images.size();
  const int npoints = input_points.front().rows();

  // Convert input images to opencv Mat
  images.resize(nimages);
  for(int i=0;i<nimages;++i) {
    images[i] = QImage2CVMat(input_images[i]);

#if 0
    // For debugging
    cv::imshow("image", images[i]);
    cv::waitKey();
#endif
  }

  // Collect all input shapes
  shapes = Eigen::MatrixXd(nimages, npoints*2);
  for(int i=0;i<nimages;++i) {
    Eigen::MatrixXd points_i_T = input_points[i].transpose();
    shapes.row(i) = Eigen::Map<Eigen::RowVectorXd>(points_i_T.data(), points_i_T.size());

#if 0
    // For debugging
    cout << shapes.row(i) << endl;
    cv::Mat img_i = images[i].clone();
    for(int j=0;j<npoints;++j) {
      cv::circle(img_i, cv::Point(shapes(i, j*2), shapes(i, j*2+1)), 1, cv::Scalar( 0, 255, 0));
    }
    cv::imshow("image", img_i);
    cv::waitKey();
#endif
  }
}

void AAMModel::BuildModel(const vector<int>& indices) {
  // Compute mean shape

  // Compute mean texture

  // Construct shape and texture model
}
