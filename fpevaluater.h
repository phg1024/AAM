#pragma once

#include "common.h"

using std::vector;
using std::string;
using std::pair;

namespace aam {

  class FeaturePointsEvaluater {
  public:
    FeaturePointsEvaluater(const vector<QImage>& images,
                           const vector<cv::Mat>& points);

    void SetOutputPath(const string& p) {
      output_path = p;
    }
    void Evaluate() const;

  protected:
    pair<vector<vector<cv::Mat>>, vector<cv::Mat>>
    ExtractFeatures(const vector<cv::Mat>& imgs,
                    const vector<cv::Mat>& pts) const;

  private:
    // Input data
    vector<cv::Mat> input_images;
    vector<cv::Mat> input_points;

    string output_path;
  };

}
