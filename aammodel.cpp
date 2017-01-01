#include "aammodel.h"
#include "utils.h"
#include "ioutils.h"

using namespace std;

using Eigen::VectorXd;
using Eigen::MatrixXd;

using cv::Mat;

AAMModel::AAMModel(const std::vector<QImage>& images, const std::vector<Mat>& points)
  : input_images(images), input_points(points) {
  Preprocess();
}

void AAMModel::Preprocess() {
  boost::timer::auto_cpu_timer t("Preprocessing finished in %w seconds.\n");

  const int nimages = input_images.size();
  const int npoints = input_points.front().rows;

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
  shapes = Mat(nimages, npoints*2, CV_64FC1);
  for(int i=0;i<nimages;++i) {
    Mat ptsT = input_points[i];
    Mat row_i = ptsT.reshape(1, 1).clone();
    row_i.copyTo(shapes.row(i));

#if 0
    // For debugging
    cout << shapes.row(i) << endl;
    Mat img_i = images[i].clone();
    DrawShape(img_i, shapes.row(i));
    cv::imshow("image", img_i);
    cv::waitKey();
#endif
  }

  triangles = LoadTriangulation("/home/phg/Data/Multilinear/landmarks_triangulation.dat");
  // Convert to 0-based indexing
  std::for_each(triangles.begin(), triangles.end(), [](cv::Vec3i& v) { v -= cv::Vec3i(1, 1, 1); });

  // Compute mean shape
  meanshape = ComputeMeanShape(shapes);

#if 0
  // For debugging
  cout << "Drawing mesh ..." << endl;
  Mat img(250, 250, CV_8UC3, cv::Scalar(0, 0, 0));
  DrawMesh(img, triangles, meanshape);
  DrawShape(img, meanshape);
  cv::imshow("meanshape", img);
  cv::waitKey();
#endif

  // Compute mean texture
  meantexture = ComputeMeanTexture(images, shapes, meanshape);

#if 1
  cv::namedWindow("mean texture", cv::WINDOW_NORMAL);
  Mat img = meantexture.clone();
  DrawMesh(img, triangles, meanshape);
  DrawShape(img, meanshape);
  cv::imshow("mean texture", img);
  cv::waitKey();
#endif

  // Put all texels into a Mat
  int ntexels = accumulate(pixel_counts.begin(), pixel_counts.end(), 0);
  textures = cv::Mat(nimages, ntexels, CV_64FC3);
  for(int i=0;i<nimages;++i) {
    int offset = 0;
    for(int j=0;j<pixel_counts.size();++j) {
      for(int k=0;k<pixel_coords[j].size();++k) {
        // collect the texels
        auto pix_coord = pixel_coords[j][k];
        textures.at<cv::Vec3d>(i, offset+k) = warped_images[i].at<cv::Vec3d>(pix_coord[0], pix_coord[1]);
      }
      offset += pixel_counts[j];
    }
  }
}

Mat AAMModel::AlignShape(const Mat& from_shape, const Mat& to_shape) {
  Mat aligned_shape;

#if 0
  cout << "from: " << from_shape << endl;
  cout << "to: " << to_shape << endl;
  PAUSE();
#endif

  const int npoints = from_shape.cols / 2;

  auto from_points = CVMat2Points(from_shape);
  auto to_points = CVMat2Points(to_shape);

#if 0
  Mat tform = cv::estimateRigidTransform(from_points, to_points, false);
#else
  Mat tform = EstimateRigidTransform(from_shape, to_shape);
#endif

#if 0
  cout << tform << endl;
  PAUSE();
#endif

  cv::transform(from_shape.reshape(2), aligned_shape, tform);
  aligned_shape = aligned_shape.reshape(1);

#if 0
  cout << aligned_shape << endl;
  PAUSE();
#endif


  return aligned_shape;
}

Mat AAMModel::ScaleShape(const Mat& shape, double size) {
  const int npoints = shape.cols / 2;

  Mat s = shape.clone();

  double max_x = -1e6;
  double min_x = 1e6;
  double max_y = -1e6;
  double min_y = 1e6;

  for(int j=0;j<npoints;++j) {
    double x = shape.at<double>(0, j*2);
    double y = shape.at<double>(0, j*2+1);

    max_x = max(max_x, x); min_x = min(min_x, x);
    max_y = max(max_y, y); min_y = min(min_y, y);
  }

  double center_x = 0.5 * (max_x + min_x);
  double center_y = 0.5 * (max_y + min_y);

  double xrange = max_x - min_x;
  double yrange = max_y - min_y;

  double factor = 0.95 * size / max(xrange, yrange);

  for(int j=0;j<npoints;++j) {
    double x = s.at<double>(0, j*2);
    double y = s.at<double>(0, j*2+1);

    s.at<double>(0, j*2) = (x - center_x) * factor + size * 0.5;
    s.at<double>(0, j*2+1) = (y - center_y) * factor + size * 0.5;
  }
  return s;
}

Mat AAMModel::ComputeMeanShape(const Mat& shapes) {
  const int npoints = shapes.cols / 2;
  const int nimages = images.size();

  Mat meanshape = Mat::zeros(1, npoints*2, CV_64FC1);
  for(int j=0;j<nimages;++j) meanshape += shapes.row(j);
  meanshape /= nimages;
  meanshape = ScaleShape(meanshape, 250);

  const int max_iters = 100;

  for(int iter=0;iter<max_iters;++iter) {
    Mat new_meanshape = Mat::zeros(1, npoints*2, CV_64FC1);
    for(int j=0;j<nimages;++j) {
      new_meanshape = new_meanshape + AlignShape(shapes.row(j), meanshape);
    }
    new_meanshape /= nimages;
    new_meanshape = ScaleShape(new_meanshape, 250);
    double norm = cv::norm(new_meanshape - meanshape);
    cout << "iter " << iter << ": Diff = " << norm << endl;
    meanshape = new_meanshape;

#if 0
    Mat img(250, 250, CV_8UC3, cv::Scalar(0, 0, 0));
    DrawShapeWithIndex(img, meanshape);
    cv::imshow("meanshape", img);
    cv::waitKey();
#endif

    if(norm < 1e-3) break;
  }

  return meanshape;
}

Mat AAMModel::ComputeMeanTexture(const vector<Mat>& images,
                                 const Mat& shapes,
                                 const Mat& meanshape) {

  const int nimages = images.size();
  const int npoints = meanshape.cols / 2;
  const int ntriangles = triangles.size();
  const int w = images.front().cols;
  const int h = images.front().rows;

  auto get_point = [&](const Mat& shape, int idx) {
    return cv::Point2f(shape.at<double>(0, idx*2),
                       shape.at<double>(0, idx*2+1));
  };

  vector<cv::Point2f> meanshape_verts(npoints);
  for(int i=0;i<npoints;++i) {
    meanshape_verts[i] = get_point(meanshape, i);
  }

  tforms.resize(nimages, vector<Mat>(ntriangles));
  tforms_inv.resize(nimages, vector<Mat>(ntriangles));

  for(int j=0;j<ntriangles;++j) {
    const int vj0 = triangles[j][0];
    const int vj1 = triangles[j][1];
    const int vj2 = triangles[j][2];

    cv::Point2f mv0 = meanshape_verts[vj0];
    cv::Point2f mv1 = meanshape_verts[vj1];
    cv::Point2f mv2 = meanshape_verts[vj2];

    for(int i=0;i<nimages;++i) {
      cv::Point2f v0 = get_point(shapes.row(i), vj0);
      cv::Point2f v1 = get_point(shapes.row(i), vj1);
      cv::Point2f v2 = get_point(shapes.row(i), vj2);

      tforms[i][j] = cv::getAffineTransform(vector<cv::Point2f>{v0, v1, v2},
                                            vector<cv::Point2f>{mv0, mv1, mv2});
      cv::invertAffineTransform(tforms[i][j], tforms_inv[i][j]);
    }
  }

  // Create pixel map in the texture space
  const int tri_id_offset = 128;
  pixel_map = Mat(h, w, CV_8UC1, cv::Scalar(0));
  for(int j=0;j<ntriangles;++j) {
    const int vj0 = triangles[j][0];
    const int vj1 = triangles[j][1];
    const int vj2 = triangles[j][2];

    FillTriangle(pixel_map, meanshape_verts[vj0], meanshape_verts[vj1], meanshape_verts[vj2], cv::Scalar(j+tri_id_offset));
  }

#if 0
  cv::imshow("pixel map", pixel_map);
  cv::waitKey();
#endif

  // Count the number of pixels we need to process
  pixel_counts.resize(ntriangles, 0);
  pixel_coords.resize(ntriangles);
  for(int i=0;i<h;++i) {
    for(int j=0;j<w;++j) {
      int tri_id = static_cast<int>(pixel_map.at<unsigned char>(i, j)) - tri_id_offset;
      if(tri_id >= 0) {
        ++pixel_counts[tri_id];
        pixel_coords[tri_id].push_back(cv::Vec2i(i, j));
      }
    }
  }

  // Create the list of points we need to project back
  pixel_mats.resize(ntriangles);
  for(int j=0;j<ntriangles;++j) {
    pixel_mats[j] = cv::Mat(pixel_counts[j], 2, CV_32FC1);
    for(int k=0;k<pixel_counts[j];++k) {
      auto pix_coord = pixel_coords[j][k];
      pixel_mats[j].at<float>(k, 0) = pix_coord[1];
      pixel_mats[j].at<float>(k, 1) = pix_coord[0];
    }
  }

  // Warp the input images to the meanshape space
  warped_images.resize(nimages);

  for(int i=0;i<nimages;++i) {
    warped_images[i] = Mat(h, w, CV_64FC3, cv::Scalar(0, 0, 0));
    for(int j=0;j<ntriangles;++j) {
      // project back the points to input image space
      cv::Mat pts;
      cv::transform(pixel_mats[j].reshape(2), pts, tforms_inv[i][j]);
      pts = pts.reshape(1, 1);

      for(int k=0;k<pixel_counts[j];++k) {
        auto pix_coord = pixel_coords[j][k];

        cv::Vec3d sample = SampleImage(images[i], cv::Point2f(pts.at<float>(0,k*2), pts.at<float>(0,k*2+1)));

        warped_images[i].at<cv::Vec3d>(pix_coord[0], pix_coord[1])
          = sample;
      }
    }

#if 0
    cv::imshow("warped", warped_images[i]);
    cv::waitKey(25);
#endif
  }

  Mat meantexture(h, w, CV_64FC3, cv::Scalar(0, 0, 0));
  for(int i=0;i<nimages;++i) {
    meantexture += warped_images[i];
#if 0
    cout << i << endl;
    cv::imshow("warped", warped_images[i]);
    cv::imshow("meantexture", meantexture / (i+1));
    cv::waitKey();
#endif
  }
  meantexture /= nimages;

  return meantexture;
}

void AAMModel::BuildModel(vector<int> indices) {
  if(indices.empty()) {
    indices.resize(input_images.size());
    std::iota(indices.begin(), indices.end(), 0);
  }

  int nimages = indices.size();

  // Construct shape and texture model with the provided indices
  cv::PCA shape_model, texture_model;
  {
    boost::timer::auto_cpu_timer t("AAM model constructed in %w seconds.\n");
    shape_model = shape_model(shapes, Mat(), CV_PCA_DATA_AS_ROW, 0.98);
    texture_model = texture_model(textures.reshape(1), Mat(), CV_PCA_DATA_AS_ROW, 0.98);
  }

#if 1
  cout << textures.rows << 'x' << textures.cols << endl;
  cout << texture_model.mean.rows << 'x' << texture_model.mean.cols << endl;

  vector<double> diffs(nimages);
  vector<Mat> reconstructions(nimages);
  for(int i=0;i<nimages;++i) {
    Mat coeffs(1, texture_model.mean.cols, texture_model.mean.type()), reconstructed;
    Mat vec = textures.row(i);
    texture_model.project(vec.reshape(1), coeffs);
    texture_model.backProject(coeffs, reconstructed);
    reconstructed = reconstructed.reshape(3);
    diffs[i] = cv::norm(vec, reconstructed, cv::NORM_L2);
    reconstructions[i] = reconstructed;
    printf("%d. diff = %g\n", i, diffs[i]);
  }

  auto max_it = max_element(diffs.begin(), diffs.end());
  auto max_idx = distance(diffs.begin(), max_it);

  cout << "max idx = " << max_idx << endl;

    // Fill the image
  auto fill_image = [&](const Mat& tex, Mat& img) {
    for(int j=0, offset=0;j<pixel_coords.size();++j) {
      for(int k=0; k<pixel_coords[j].size(); ++k) {
        auto pix = pixel_coords[j][k];
        img.at<cv::Vec3d>(pix[0], pix[1]) = tex.at<cv::Vec3d>(0, offset+k);
      }
      offset += pixel_counts[j];
    }
  };

  cv::Mat img(images.front().rows, images.front().cols, images.front().type(), cv::Scalar(0, 0, 0));
  fill_image(reconstructions[max_idx], img);
  cv::imshow("outlier", img);

  cv::Mat img_ref(images.front().rows, images.front().cols, images.front().type(), cv::Scalar(0, 0, 0));
  fill_image(textures.row(max_idx), img_ref);
  cv::imshow("ref", img_ref);
  cv::waitKey();
#endif
}
