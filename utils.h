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

inline void DrawShape(cv::Mat& img, const cv::Mat& shape) {
  const int npoints = shape.cols / 2;
  for(int j=0;j<npoints;++j) {
    cv::circle(img, cv::Point(shape.at<double>(0, j*2), shape.at<double>(0, j*2+1)), 1, cv::Scalar( 0, 255, 0));
  }
}

inline void DrawShapeWithIndex(cv::Mat& img, const cv::Mat& shape) {
  const int npoints = shape.cols / 2;
  for(int j=0;j<npoints;++j) {
    auto p_j = cv::Point(shape.at<double>(0, j*2), shape.at<double>(0, j*2+1));
    cv::circle(img, p_j, 1, cv::Scalar( 0, 255, 0));
    cv::putText(img, std::to_string(j), p_j, cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(255, 175, 175));
  }
}

inline void DrawMesh(cv::Mat& img, const std::vector<cv::Vec3i>& faces, const cv::Mat& verts) {
  auto get_point = [&](int idx) {
    return cv::Point(verts.at<double>(idx*2), verts.at<double>(idx*2+1));
  };

  std::cout << faces.size() << " faces." << std::endl;
  for(auto f : faces) {
    auto v0 = get_point(f[0]);
    auto v1 = get_point(f[1]);
    auto v2 = get_point(f[2]);

    line(img, v0, v1, cv::Scalar(255, 175, 175), 1, CV_AA);
    line(img, v1, v2, cv::Scalar(255, 175, 175), 1, CV_AA);
    line(img, v2, v0, cv::Scalar(255, 175, 175), 1, CV_AA);
  }
}

inline void FillTriangle(cv::Mat& img, const cv::Point2f& v0, const cv::Point2f& v1, const cv::Point2f& v2, cv::Scalar s) {
  cv::Point pts[] = {v0, v1, v2};
  cv::fillConvexPoly(img, pts, 3, s);
}

inline void PAUSE() { getchar(); }

inline std::vector<cv::Point2f> CVMat2Points(const cv::Mat& m) {
  const int npoints = m.cols/2;
  std::vector<cv::Point2f> points(npoints);
  for(int i=0;i<npoints;++i) {
    points[i] = cv::Point2f(m.at<double>(0,i*2), m.at<double>(0, i*2+1));
  }
  return points;
}

inline Eigen::VectorXd CVMat2EigenVec(const cv::Mat& m) {
  Eigen::VectorXd v(m.cols);
  for(int i=0;i<m.cols;++i) v(i) = m.at<double>(0, i);
  return v;
}

inline cv::Mat EstimateRigidTransform(const cv::Mat& from_shape,
                                      const cv::Mat& to_shape) {
  Eigen::VectorXd p = CVMat2EigenVec(from_shape);
  Eigen::VectorXd q = CVMat2EigenVec(to_shape);

  assert(p.rows() == q.rows());

  int n = p.rows() / 2;
  assert(n>0);
  const int m = 2;

  Eigen::Map<const Eigen::MatrixXd> pmatT(p.data(), 2, n);
  Eigen::Map<const Eigen::MatrixXd> qmatT(q.data(), 2, n);

  Eigen::MatrixXd pmat = pmatT.transpose();
  Eigen::MatrixXd qmat = qmatT.transpose();

  Eigen::MatrixXd mu_p = pmat.colwise().mean();
  Eigen::MatrixXd mu_q = qmat.colwise().mean();

  Eigen::MatrixXd dp = pmat - mu_p.replicate(n, 1);
  Eigen::MatrixXd dq = qmat - mu_q.replicate(n, 1);

  double sig_p2 = dp.squaredNorm() / n;
  double sig_q2 = dq.squaredNorm() / n;

  Eigen::MatrixXd sig_pq = dq.transpose() * dp / n;

  double det_sig_pq = sig_pq.determinant();
  Eigen::MatrixXd S = Eigen::MatrixXd::Identity(m, m);
  if (det_sig_pq < 0) S(m - 1, m - 1) = -1;

  Eigen::MatrixXd U, V;
  Eigen::VectorXd D;

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(sig_pq, Eigen::ComputeFullU | Eigen::ComputeFullV);
  D = svd.singularValues();
  U = svd.matrixU();
  V = svd.matrixV();

  Eigen::MatrixXd R = U * S * V.transpose();

  Eigen::Matrix2d Dmat;
  Dmat << D[0], 0, 0, D[1];

  double s = (Dmat * S).trace() / sig_p2;

  Eigen::VectorXd t = mu_q.transpose() - s * R * mu_p.transpose();

  R = R * s;

  cv::Mat tform(2, 3, CV_64FC1);
  tform.at<double>(0, 0) = R(0, 0); tform.at<double>(0, 1) = R(0, 1);
  tform.at<double>(1, 0) = R(1, 0); tform.at<double>(1, 1) = R(1, 1);

  tform.at<double>(0, 2) = t(0); tform.at<double>(1, 2) = t(1);
  return tform;
}

inline cv::Vec3d SampleImage(const cv::Mat& I, const cv::Point2f& p) {
  int x0 = p.x, y0 = p.y;
  int x1 = x0 + 1, y1 = y0 + 1;

  //bilinear interpolation
  float dx = p.x - x0;
  float dy = p.y - y0;

  return I.at<cv::Vec3d>(y0, x0) * (1.0 - dx) * (1.0 - dy)
       + I.at<cv::Vec3d>(y0, x1) * (dx)       * (1.0 - dy)
       + I.at<cv::Vec3d>(y1, x0) * (1.0 - dx) * (dy)
       + I.at<cv::Vec3d>(y1, x1) * (dx)       * (dy);
}

inline void FillImage(const cv::Mat& tex,
                       const std::vector<std::vector<cv::Vec2i>>& pixel_coords,
                       cv::Mat& img) {
  for(int j=0, offset=0;j<pixel_coords.size();++j) {
    for(int k=0; k<pixel_coords[j].size(); ++k) {
      auto pix = pixel_coords[j][k];
      img.at<cv::Vec3d>(pix[0], pix[1]) = tex.at<cv::Vec3d>(0, offset+k);
    }
    offset += pixel_coords[j].size();
  }
};
