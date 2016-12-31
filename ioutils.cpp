#include "ioutils.h"

using namespace std;

vector<string> ReadFileByLine(const string &filename) {
  ifstream fin(filename);
  vector<string> lines;
  while (fin) {
    string line;
    std::getline(fin, line);
    if (!line.empty())
      lines.push_back(line);
  }
  return lines;
}

vector<pair<string, string>> ParseSettingsFile(const string& filename) {
  vector<string> lines = ReadFileByLine(filename);

  vector<pair<string, string>> image_points_filenames(lines.size());
  std::transform(lines.begin(), lines.end(), image_points_filenames.begin(),
                 [](const string &line) {
                   vector<string> parts;
                   boost::algorithm::split(parts, line,
                                           boost::algorithm::is_any_of(" "),
                                           boost::algorithm::token_compress_on);
                   auto parts_end = std::remove_if(parts.begin(), parts.end(),
                                                   [](const string &s) {
                                                     return s.empty();
                                                   });
                   assert(std::distance(parts.begin(), parts_end) == 2);
                   return make_pair(parts.front(), parts.back());
                 });
  return image_points_filenames;
}

Eigen::MatrixX2d ReadPoints(const string& filename) {
  ifstream fin(filename);
  int npoints;
  fin >> npoints;
  cout << "reading " << npoints << " points ..." << endl;
  Eigen::MatrixX2d pts(npoints, 2);
  for(int i=0;i<npoints;++i) {
    fin >> pts(i, 0) >> pts(i, 1);
  }

  return pts;
}

pair<QImage, Eigen::MatrixX2d> LoadImagePointsPair(
  const string& image_filename,
  const string& points_filename
) {
  QImage img(image_filename.c_str());
  cout << "image size: " << img.width() << "x" << img.height() << endl;

  Eigen::MatrixX2d pts = ReadPoints(points_filename);
  cout << "number of points: " << pts.rows() << endl;

  // Convert points to 0-based coordinates
  pts -= Eigen::MatrixX2d::Ones(pts.rows(), 2);

  return make_pair(img, pts);
}
