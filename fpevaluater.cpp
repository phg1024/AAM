#include "fpevaluater.h"
#include "utils.h"

#include "features/vl_hog.h"

using namespace std;
using namespace cv;

namespace aam {

namespace {
  /**
   * Function object that extracts HoG features at given 2D landmark locations
   * and returns them as a row vector.
   *
   * We wrap all the C-style memory allocations of the VLFeat library
   * in cv::Mat's.
   * Note: Any other library and features can of course be used.
   */
  class HogTransform
  {
  public:
  	HogTransform(VlHogVariant vlhog_variant, int num_cells, int cell_size, int num_bins) : vlhog_variant(vlhog_variant), num_cells(num_cells), cell_size(cell_size), num_bins(num_bins)
  	{
  	};

  	pair<vector<cv::Mat>, cv::Mat> operator()(const cv::Mat& img, const cv::Mat& pts)
  	{
      //cout << pts.rows << "x" << pts.cols << endl;
      //cout << img.rows << "x" << img.cols << "x" << img.channels() << endl;
  		assert(pts.cols == 2);
  		using cv::Mat;

  		Mat gray_image;
  		if (img.channels() == 3) {
  			cv::cvtColor(img, gray_image, cv::COLOR_BGR2GRAY);
  		}
  		else {
  			gray_image = img;
  		}

  		// Note: We could use the 'regressorLevel' to choose the window size (and
  		// other parameters adaptively). We omit this for the sake of a short example.

  		int patch_width_half = num_cells * (cell_size / 2);

  		Mat hog_descriptors; // We'll get the dimensions later from vl_hog_get_*

  		const int num_landmarks = pts.rows;
      vector<cv::Mat> patches;
  		for (int i = 0; i < num_landmarks; ++i) {
  			int x = cvRound(pts.at<double>(i, 0));
  			int y = cvRound(pts.at<double>(i, 1));

  			Mat roi_img;
  			if (x - patch_width_half < 0 || y - patch_width_half < 0 || x + patch_width_half >= gray_image.cols || y + patch_width_half >= gray_image.rows) {
  				// The feature extraction location is too far near a border. We extend the
  				// image (add a black canvas) and then extract from this larger image.
  				int borderLeft = (x - patch_width_half) < 0 ? std::abs(x - patch_width_half) : 0; // x and y are patch-centers
  				int borderTop = (y - patch_width_half) < 0 ? std::abs(y - patch_width_half) : 0;
  				int borderRight = (x + patch_width_half) >= gray_image.cols ? std::abs(gray_image.cols - (x + patch_width_half)) : 0;
  				int borderBottom = (y + patch_width_half) >= gray_image.rows ? std::abs(gray_image.rows - (y + patch_width_half)) : 0;
  				Mat extendedImage = gray_image.clone();
  				cv::copyMakeBorder(extendedImage, extendedImage, borderTop, borderBottom, borderLeft, borderRight, cv::BORDER_CONSTANT, cv::Scalar(0));
  				cv::Rect roi((x - patch_width_half) + borderLeft, (y - patch_width_half) + borderTop, patch_width_half * 2, patch_width_half * 2); // Rect: x y w h. x and y are top-left corner.
  				roi_img = extendedImage(roi).clone(); // clone because we need a continuous memory block
  			}
  			else {
  				cv::Rect roi(x - patch_width_half, y - patch_width_half, patch_width_half * 2, patch_width_half * 2); // x y w h. Rect: x and y are top-left corner. Our x and y are center. Convert.
  				roi_img = gray_image(roi).clone(); // clone because we need a continuous memory block
  			}
        //cout << roi_img.rows << 'x' << roi_img.cols << endl;
        //cv::imshow("roi", roi_img);
        //cv::waitKey();
        patches.push_back(roi_img);

  			roi_img.convertTo(roi_img, CV_32FC1); // vl_hog_put_image expects a float* (values 0.0f-255.0f)
  			VlHog* hog = vl_hog_new(vlhog_variant, num_bins, false); // transposed (=col-major) = false
  			vl_hog_put_image(hog, (float*)roi_img.data, roi_img.cols, roi_img.rows, 1, cell_size); // (the '1' is numChannels)
  			int ww = static_cast<int>(vl_hog_get_width(hog)); // assert ww == hh == numCells
  			int hh = static_cast<int>(vl_hog_get_height(hog));
  			int dd = static_cast<int>(vl_hog_get_dimension(hog)); // assert ww=hogDim1, hh=hogDim2, dd=hogDim3
        //cout << ww << 'x' << hh << 'x' << dd << endl;
  			Mat hogArray(1, ww*hh*dd, CV_32FC1); // safer & same result. Don't use C-style memory management.
  			vl_hog_extract(hog, hogArray.ptr<float>(0));
  			vl_hog_delete(hog);
  			Mat hogDescriptor(hh*ww*dd, 1, CV_32FC1);
  			// Stack the third dimensions of the HOG descriptor of this patch one after each other in a column-vector:
  			for (int j = 0; j < dd; ++j) {
  				Mat hogFeatures(hh, ww, CV_32FC1, hogArray.ptr<float>(0) + j*ww*hh); // Creates the same array as in Matlab. I might have to check this again if hh!=ww (non-square)
  				hogFeatures = hogFeatures.t(); // necessary because the Matlab reshape() takes column-wise from the matrix while the OpenCV reshape() takes row-wise.
  				hogFeatures = hogFeatures.reshape(0, hh*ww); // make it to a column-vector
  				Mat currentDimSubMat = hogDescriptor.rowRange(j*ww*hh, j*ww*hh + ww*hh);
  				hogFeatures.copyTo(currentDimSubMat);
  			}
  			hogDescriptor = hogDescriptor.t(); // now a row-vector
  			hog_descriptors.push_back(hogDescriptor);
  		}
  		// concatenate all the descriptors for this sample vertically (into a row-vector):
  		hog_descriptors = hog_descriptors.reshape(0, hog_descriptors.cols * num_landmarks).t();
  		return make_pair(patches, hog_descriptors);
  	};

  private:
  	VlHogVariant vlhog_variant;
  	int num_cells;
  	int cell_size;
  	int num_bins;
  };
}

FeaturePointsEvaluater::FeaturePointsEvaluater(
  const vector<QImage>& images,
  const vector<cv::Mat>& points) {
    input_images.resize(images.size());
    for(int i=0;i<images.size();++i) input_images[i] = QImage2CVMatU(images[i]);
    input_points = points;
}

void FeaturePointsEvaluater::Evaluate() const {
  cout << "Extracing features ..." << endl;
  vector<Mat> features;
  vector<vector<Mat>> patches;
  tie(patches, features) = ExtractFeatures(input_images, input_points);

  {
    ofstream fout("features.txt");
    for(int i=0;i<features.size();++i) {
      cout << "Features: " << features[i].rows << "x" << features[i].cols << endl;
      fout << features[i] << endl;
    }
    fout.close();
  }

  {
    const int nimages = patches.size();
    const int npoints = patches.front().size();
    for(int i=0;i<npoints;++i) {
      for(int j=0;j<nimages;++j) {
        string patch_filename = "patch_" + to_string(i) + "_" + to_string(j) + ".png";
        cv::imwrite(output_path + "/" + patch_filename, patches[j][i]);
      }
    }
  }

  bool use_patch = false;
  if(use_patch){
    const int nimages = patches.size();
    const int npoints = patches.front().size();
    const int patch_size = patches[0][0].rows * patches[0][0].cols;
    vector<Mat> patches_db(npoints);
    for(int i=0;i<npoints;++i) {
      patches_db[i] = Mat(nimages, patch_size, CV_32FC1);
      for(int j=0;j<nimages;++j) {
        Mat patch_ji;
        patches[j][i].convertTo(patch_ji, CV_32F);

        //cv::imshow("patch " + to_string(i) + "_" + to_string(j), patches[j][i]);
        //cv::waitKey();

        patches_db[i].row(j) = (patch_ji.reshape(0, 1) / 255.0);
      }
      //cout << patches_db[i] << endl;
    }

    cout << cv::norm(patches_db[0], patches_db[1], cv::NORM_L2) << endl;

    // Construct PCA model for each patch
    vector<cv::PCA> patch_models(npoints);
    vector<pair<int, double>> error(nimages);
    for(int i=0;i<nimages;++i) error[i] = make_pair(i, 0);

    for(int i=0;i<npoints;++i) {
      cout << "patch " << i << endl;
      // construct PCA model
      patch_models[i] = patch_models[i](patches_db[i], Mat(), CV_PCA_DATA_AS_ROW, 0.5);

      // reconstruct patch
      for(int j=0;j<nimages;++j) {
        Mat coeffs(1, patch_models[i].mean.cols, patch_models[i].mean.type());
        patch_models[i].project(patches_db[i].row(j), coeffs);
        Mat reconstructed;
        patch_models[i].backProject(coeffs, reconstructed);
        //cout << patches_db[i].row(j).rows << 'x' << patches_db[i].row(j).cols << endl;
        //cout << reconstructed.rows << 'x' << reconstructed.cols << endl;
        error[j].second += cv::norm(patches_db[i].row(j), reconstructed, cv::NORM_L2);

        // save it
        string patch_filename = "patch_" + to_string(i) + "_" + to_string(j) + "_recon" + ".png";
        cv::imwrite(output_path + "/" + patch_filename, reconstructed.reshape(0, 16) * 255.0);
      }
    }

    sort(error.begin(), error.end(), [](const pair<int, double>& a, const pair<int, double>& b) {
      return a.second > b.second;
    });
    for(int i=0;i<nimages;++i) {
      cout << "(" << error[i].first << ", " << error[i].second << ") "; cout << endl;
      Mat img = input_images[error[i].first].clone();
      DrawShape(img, input_points[error[i].first].reshape(0, 1));
      cv::imshow("img", img);
      cv::waitKey();
    }
  }
  else {
    const int nimages = patches.size();
    const int npoints = patches.front().size();
    const int patch_size = 128;
    vector<Mat> patches_db(npoints);
    for(int i=0;i<npoints;++i) {
      patches_db[i] = Mat(nimages, patch_size, CV_32FC1);
      for(int j=0;j<nimages;++j) {
        Mat patch_ji = features[j](cv::Range::all(), cv::Range(i*patch_size, (i+1)*patch_size)).clone();
        //cout << patch_ji.rows << 'x' << patch_ji.cols << ": " << patch_ji << endl;
        //cv::imshow("patch " + to_string(i) + "_" + to_string(j), patches[j][i]);
        //cv::waitKey();

        patch_ji.copyTo(patches_db[i].row(j));
      }
      //cout << patches_db[i] << endl;
    }

    // Construct PCA model for each patch
    vector<cv::PCA> patch_models(npoints);
    vector<pair<int, double>> error(nimages);
    for(int i=0;i<nimages;++i) error[i] = make_pair(i, 0);

    for(int i=0;i<npoints;++i) {
      //cout << "patch " << i << endl;
      // construct PCA model
      patch_models[i] = patch_models[i](patches_db[i], Mat(), CV_PCA_DATA_AS_ROW, 0.5);

      // reconstruct patch
      for(int j=0;j<nimages;++j) {
        Mat coeffs(1, patch_models[i].mean.cols, patch_models[i].mean.type());
        patch_models[i].project(patches_db[i].row(j), coeffs);
        Mat reconstructed;
        patch_models[i].backProject(coeffs, reconstructed);
        //cout << patches_db[i].row(j).rows << 'x' << patches_db[i].row(j).cols << endl;
        //cout << reconstructed.rows << 'x' << reconstructed.cols << endl;
        error[j].second += cv::norm(patches_db[i].row(j), reconstructed, cv::NORM_L2);
      }
    }

    sort(error.begin(), error.end(), [](const pair<int, double>& a, const pair<int, double>& b) {
      return a.second > b.second;
    });
    for(int i=0;i<nimages;++i) {
      cout << "(" << error[i].first << ", " << error[i].second << ") "; cout << endl;
      Mat img = input_images[error[i].first].clone();
      DrawShape(img, input_points[error[i].first].reshape(0, 1));
      cv::imshow("img", img);
      cv::waitKey();
    }
  }
}

pair<vector<vector<Mat>>, vector<Mat>> FeaturePointsEvaluater::ExtractFeatures(const vector<Mat>& images,
                                                    const vector<Mat>& points) const{
  HogTransform hog(VlHogVariant::VlHogVariantDalalTriggs, 4/*numCells*/, 4 /*cellSize*/, 2 /*numBins*/);

  vector<Mat> features(images.size());
  vector<vector<Mat>> patches(images.size());
  for(int i=0;i<images.size();++i) {
    tie(patches[i], features[i]) = hog(images[i], points[i]);
  }

  return make_pair(patches, features);
}

}
