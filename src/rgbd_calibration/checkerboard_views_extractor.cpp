/*
 *  Copyright (c) 2013-2014, Filippo Basso <bassofil@dei.unipd.it>
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *     1. Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *     2. Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *     3. Neither the name of the copyright holder(s) nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <rgbd_calibration/checkerboard_views_extractor.h>

#include <calibration_common/algorithms/plane_extraction.h>
#include <calibration_common/algorithms/interactive_checkerboard_finder.h>
#include <calibration_common/algorithms/automatic_checkerboard_finder.h>

namespace calibration
{

  /**
   * @brief The AllPointPlaneExtractor class
   * @param PCLPointT_
   */
  template <typename PCLPointT_>
  class AllPointPlaneExtraction : public PointPlaneExtraction<PCLPointT_>
  {
  public:
    virtual bool extract(PlaneInfo & plane_info) const;
  };


Size1 CheckerboardViewsExtraction::extract(const RGBDData::ConstPtr & data,
                                           std::vector<CheckerboardViews::Ptr> & cb_views_vec,
                                           bool interactive,
                                           bool force,
                                           bool assumeAllOnPlane) const
{
  Size1 added = 0;

  cv::Mat image = data->colorData().clone();
  AutomaticCheckerboardFinder finder;
  finder.setImage(image);

  PointPlaneExtraction<PCLPoint3>::Ptr plane_extractor;

  if (assumeAllOnPlane)
    plane_extractor = boost::make_shared<AllPointPlaneExtraction<PCLPoint3> >();
  else if (not interactive)
    plane_extractor = boost::make_shared<PointPlaneExtraction<PCLPoint3> >();
  else
    plane_extractor = boost::make_shared<PointPlaneExtractionGUI<PCLPoint3> >();

  plane_extractor->setInputCloud(data->depthData());

  for (Size1 c = 0; c < cb_vec_.size(); ++c)
  {
    const Checkerboard::ConstPtr & cb = cb_vec_[c];

    plane_extractor->setRadius(std::min(cb->width(), cb->height()) / 1.5);

    std::stringstream ss;
    ss << "rgbd_cb_" << data->id() << "_" << c;
    CheckerboardViews::Ptr cb_views(boost::make_shared<CheckerboardViews>(ss.str()));
    cb_views->setData(data);
    cb_views->setCheckerboard(cb);

    // 1. Extract corners

    Cloud2 image_corners(cb->corners().size());
    if (not finder.find(*cb, image_corners))
    {
      if (force)
      {
        InteractiveCheckerboardFinder finder2;
        finder2.setImage(image);
        if (not finder2.find(*cb, image_corners))
          continue;
      }
      else
        continue;
    }

    cb_views->setImageCorners(image_corners);

    if (not cb_constraint_->isValid(*cb_views->colorCheckerboard()))
      continue;

    // 2. Extract plane

    if (not only_images_)
    {
      PlaneInfo plane_info;

      if (interactive)
      {
        cb_views->draw(image);
      }
      else
      {
        Point3 center = color_sensor_pose_ * cb_views->colorCheckerboard()->center();
        PCLPoint3 p;
        p.x = center[0];
        p.y = center[1];
        p.z = center[2];
        plane_extractor->setPoint(p);
      }

      if (not plane_extractor->extract(plane_info))
        continue;

      cb_views->setPlaneInliers(plane_info.indices_, plane_info.std_dev_);

      if (not plane_constraint_->isValid(*cb_views->depthPlane()))
        continue;

    }

#pragma omp critical
    cb_views_vec.push_back(cb_views);

    ++added;
  }

  return added;

}

Size1 CheckerboardViewsExtraction::extract(std::vector<CheckerboardViews::Ptr> & cb_views_vec,
                                            bool interactive,
                                            bool assumeAllOnPlane) const
{
  return extract(data_, cb_views_vec, interactive, true, assumeAllOnPlane);
}

Size1 CheckerboardViewsExtraction::extractAll(std::vector<CheckerboardViews::Ptr> & cb_views_vec,
                                              bool interactive,
                                              bool assumeAllOnPlane) const
{
  Size1 added = 0;

#pragma omp parallel for
  for (Size1 i = 0; i < data_vec_.size(); ++i)
  {
    Size1 n = extract(data_vec_[i], cb_views_vec, interactive, force_, assumeAllOnPlane);
#pragma omp atomic
    added += n;
  }

  return added;
}

template <typename PointT_>
bool AllPointPlaneExtraction<PointT_>::extract(PlaneInfo & plane_info) const
{
  size_t nPoints = PointPlaneExtraction<PointT_>::cloud_->size();

  std::vector<int> all_indices(nPoints);
  for (size_t i = 0; i < nPoints; ++i)
    all_indices[i] = static_cast<int>(i);

  plane_info.indices_ = boost::make_shared<std::vector<int> >(all_indices);

  // From PointPlaneExtractor::extract;
  Eigen::VectorXf coefficients(Eigen::VectorXf::Zero(4));
  boost::shared_ptr<typename PointPlaneExtraction<PointT_>::ModelNormalPlane> model = boost::make_shared<typename PointPlaneExtraction<PointT_>::ModelNormalPlane>(PointPlaneExtraction<PointT_>::cloud_);
  model->setInputNormals(PointPlaneExtraction<PointT_>::cloud_normals_);
  model->setIndices(plane_info.indices_);

  model->setNormalDistanceWeight(0.0);

  Eigen::VectorXf new_coefficients(4);
  model->optimizeModelCoefficients(*plane_info.indices_, coefficients, new_coefficients);
  if (coefficients == new_coefficients)
    return false;
  coefficients = new_coefficients;


  std::vector<double> distances;
  model->getDistancesToModel(coefficients, distances);

  Eigen::Map<Eigen::VectorXd> v = Eigen::Map<Eigen::VectorXd>(&distances[0], distances.size());
  double v_mean = v.mean();
  plane_info.std_dev_ = std::sqrt(v.cwiseAbs2().mean() - v_mean * v_mean);

//    std::cout << "plane_info.std_dev_: " << plane_info.std_dev_ << std::endl;

  model->setIndices(all_indices);
  model->setAxis(Eigen::Vector3f(coefficients[0], coefficients[1], coefficients[2]));
  model->setEpsAngle(10.0);
  pcl::RandomSampleConsensus<PointT_> ransac(model);
  ransac.setDistanceThreshold(/*std::max(0.15, */10 * plane_info.std_dev_/*)*/);
  ransac.computeModel();
  ransac.getModelCoefficients(coefficients);
  ransac.getInliers(*plane_info.indices_);

  model->setIndices(plane_info.indices_);
  distances.clear();
  model->getDistancesToModel(coefficients, distances);

  Eigen::Map<Eigen::VectorXd> v2 = Eigen::Map<Eigen::VectorXd>(&distances[0], distances.size());
  double v2_mean = v2.mean();
  plane_info.std_dev_ = std::sqrt(v2.cwiseAbs2().mean() - v2_mean * v2_mean);

  model->setIndices(all_indices);
  //model->setAxis(Eigen::Vector3f(coefficients[0], coefficients[1], coefficients[2]));
  //model->setEpsAngle(10.0);

  //std::cout << "plane_info.std_dev_: " << plane_info.std_dev_ << std::endl;

  ransac = pcl::RandomSampleConsensus<PointT_>(model);
  ransac.setDistanceThreshold(/*std::max(0.15, */6 * plane_info.std_dev_/*)*/);
  ransac.computeModel();
  ransac.getModelCoefficients(coefficients);
  ransac.getInliers(*plane_info.indices_);

  plane_info.plane_.normal()[0] = coefficients[0];
  plane_info.plane_.normal()[1] = coefficients[1];
  plane_info.plane_.normal()[2] = coefficients[2];
  plane_info.plane_.offset() = coefficients[3];

  return true;
}


} /* namespace calibration */
