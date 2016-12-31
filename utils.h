#pragma once

#include "common.h"

template <typename C>
std::vector<std::pair<int, typename C::value_type>> enumerate(const C& container) {
  std::vector<std::pair<int, typename C::value_type>> pairs;
  int idx = 0;
  for(auto it=container.begin(); it!=container.end(); ++it) {
    pairs.push_back(std::make_pair(idx++, *it));
  }
  return pairs;
}

inline cv::Mat QImage2CVMat(const QImage& img) {
  cv::Mat mat(img.width(), img.height(), CV_64FC3, cv::Scalar(0));

  for(int i=0;i<img.height();++i) {
    for(int j=0;j<img.width();++j) {
      auto pij = img.pixel(j, i);

      // OpenCV uses BGR format, while QImage is RGB
      cv::Vec3d pix(qBlue(pij) / 255.0, qGreen(pij) / 255.0, qRed(pij) / 255.0);
      mat.at<cv::Vec3d>(i, j) = pix;
    }
  }

  return mat;
}