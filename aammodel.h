#pragma once

#include "common.h"

namespace aam {
  class AAMModel {
  public:
    enum ErrorMetric {
      TextureError = 0,
      FittingError,
      Hybrid
    };

    enum Method {
      LeaveOneOut,
      RobustPCA
    };
  public:
    AAMModel();
    AAMModel(const std::vector<QImage>& images, const std::vector<cv::Mat>& points);
    ~AAMModel(){}

    void SetImages(const std::vector<QImage>& images);
    void SetPoints(const std::vector<cv::Mat>& points);
    void SetOutputPath(const std::string& path);
    void SetErrorMetric(ErrorMetric m) {
      metric = m;
    }
    void SetThreshold(double val) {
      threshold = val;
    }

    void Preprocess();
    void ProcessImages();
    void ProcessShapes();
    void InitializeMeanShapeAndTexture();

    void BuildModel(std::vector<int> indices = std::vector<int>());
    std::vector<int> FindInliers_Iterative(std::vector<int> indices = std::vector<int>(), Method method = RobustPCA);

    std::vector<int> FindInliers(std::vector<int> indices = std::vector<int>());
    std::vector<int> FindInliers_RPCA(std::vector<int> indices = std::vector<int>());

  protected:
    void Init();

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

    // Output related
    std::string output_path;

    // Converted data
    std::vector<cv::Mat> images, warped_images;
    cv::Mat shapes;
    cv::Mat textures, normalized_textures;

    std::vector<cv::Vec3i> triangles; //!< triangulation of the shapes

    // Affine transformation from each triangle to the meanshape in each image
    std::vector<std::vector<cv::Mat>> tforms, tforms_inv;

    const int tri_id_offset = 128;
    cv::Mat pixel_map;  //!< pixel to triangle indices map in the texture space
    std::vector<cv::Mat> inv_pixel_maps;  //!< pixel to triangle indices map in the input image space

    std::vector<int> pixel_counts;  //!< pixel counts in texture space
    std::vector<std::vector<int>> inv_pixel_counts; //!< pixel counts in image space

    std::vector<std::vector<cv::Vec2i>> pixel_coords;
    std::vector<std::vector<std::vector<cv::Vec2i>>> inv_pixel_coords;
    std::vector<cv::Mat> pixel_mats;
    std::vector<std::vector<cv::Mat>> inv_pixel_mats, inv_pixel_pts;

    cv::Mat meanshape, meantexture;

    ErrorMetric metric;
    double threshold;
  };
}
