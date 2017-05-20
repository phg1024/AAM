#include "common.h"
#include "ioutils.h"
#include "fpevaluater.h"
#include "utils.h"

using namespace std;
using namespace cv;
using namespace aam;

int main(int argc, char** argv) {

  namespace fs = boost::filesystem;
  namespace po = boost::program_options;

  po::options_description desc("Options");
  desc.add_options()
    ("settings_file", po::value<string>()->required(), "Input settings file")
    ("output_path", po::value<string>()->default_value("."), "Output folder")
    ("mode", po::value<string>()->default_value("filter"), "Mode to run")
    ("threshold", po::value<double>()->default_value(2.0), "Threshold for outlier");

  po::variables_map vm;

  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if(vm.count("help")) {
      cout << desc << endl;
      return 1;
    }

    // nothing to do after successful parsing command line arguments

  } catch(po::error& e) {
    cerr << "Error: " << e.what() << endl;
    cerr << desc << endl;
    return 1;
  }

  const string settings_filename(vm["settings_file"].as<string>());

  // Parse the setting file and load image related resources
  fs::path settings_filepath(settings_filename);

  vector<pair<string, string>> image_points_filenames = ParseSettingsFile(settings_filename);

  int nentries = image_points_filenames.size();
  vector<QImage> images(nentries);
  vector<cv::Mat> points(nentries);

  for(auto& p : enumerate(image_points_filenames)) {
    fs::path image_filename = settings_filepath.parent_path() / fs::path(p.second.first);
    fs::path pts_filename = settings_filepath.parent_path() / fs::path(p.second.second);
    cout << "[" << image_filename << ", " << pts_filename << "]" << endl;

    int i = p.first;
    tie(images[i], points[i]) = LoadImagePointsPair(image_filename.string(), pts_filename.string());
  }

  FeaturePointsEvaluater eval(images, points);
  eval.SetOutputPath(vm["output_path"].as<string>());
  eval.Evaluate();

  return 0;
}
