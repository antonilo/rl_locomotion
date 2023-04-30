//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <deque>
#include <stdlib.h>
#include <math.h>
#include <cstdint>
#include <set>
#include <vector>
#include <string>
#include <random>
#include "../../RaisimGymEnv.hpp"
#include <fenv.h>
#include <stdexcept>

template <typename T>
inline void scaling(T &n, const T &max_src, const T &max_dst)
{
  n = n / max_src * max_dst;
}

inline void quatToEuler(const raisim::Vec<4> &quat, Eigen::Vector3d &eulerVec)
{
  double qw = quat[0], qx = quat[1], qy = quat[2], qz = quat[3];
  // roll (x-axis rotation)
  double sinr_cosp = 2 * (qw * qx + qy * qz);
  double cosr_cosp = 1 - 2 * (qx * qx + qy * qy);
  eulerVec[0] = std::atan2(sinr_cosp, cosr_cosp);

  // pitch (y-axis rotation)
  double sinp = 2 * (qw * qy - qz * qx);
  if (std::abs(sinp) >= 1)
    eulerVec[1] = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
  else
    eulerVec[1] = std::asin(sinp);

  // yaw (z-axis rotation)
  double siny_cosp = 2 * (qw * qz + qx * qy);
  double cosy_cosp = 1 - 2 * (qy * qy + qz * qz);
  eulerVec[2] = std::atan2(siny_cosp, cosy_cosp);
}

inline void eulerToRot(const Eigen::Vector3d &eulerVec, Eigen::Matrix3d &rot)
{
  double roll = eulerVec[0], pitch = eulerVec[1], yaw = eulerVec[2];
  Eigen::Quaterniond q;
  q = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
  rot = q.matrix();
}

namespace raisim
{

  class ENVIRONMENT : public RaisimGymEnv
  {

  public:
    explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg, bool visualizable, int env_id) : RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable)
    {

      setSeed(env_id);
      for (int i = 0; i < 10; i++)
        clean_randomizer = Eigen::VectorXd::Random(1)[0];

      /// add objects
      READ_YAML(bool, isTest, cfg["test"])
      READ_YAML(bool, isEval, cfg["eval"])
      READ_YAML(bool, roughTerrain, cfg["roughTerrain"])
      READ_YAML(bool, privinfo, cfg["privinfo"])
      READ_YAML(bool, includeGRF, cfg["includeGRF"])
      READ_YAML(int, baseDim, cfg["baseDim"])
      READ_YAML(int, geomDim, cfg["geomDim"])
      READ_YAML(int, n_futures, cfg["n_futures"])
      READ_YAML(int, history_len, cfg["history_len"])

      READ_YAML(bool, randomize_friction, cfg["randomize_friction"])
      READ_YAML(bool, randomize_mass, cfg["randomize_mass"])
      READ_YAML(bool, randomize_motor_strength, cfg["randomize_motor_strength"])
      READ_YAML(bool, randomize_gains, cfg["randomize_gains"])
      READ_YAML(bool, randomize_yaw, cfg["randomize_yaw"])
      READ_YAML(bool, use_priv_vel, cfg["use_priv_vel"])
      READ_YAML(bool, use_slope_dots, cfg["use_slope_dots"])

      /// Target Speed
      READ_YAML(bool, cts_target_speed, cfg["cts_target_speed"])
      READ_YAML(double, target_end_speed, cfg["target_end_speed"])
      READ_YAML(double, pid_coeff, cfg["pid_coeff"])
      READ_YAML(double, slope_th, cfg["slope_threshold"])
      READ_YAML(double, target_start_speed, cfg["target_start_speed"])
      READ_YAML(double, target_speed_period, cfg["target_speed_period"])
      READ_YAML(double, bodyLinearVel_avg_weight, cfg["bodyLinearVel_avg_weight"])
      READ_YAML(bool, speedTest, cfg["speedTest"])
      READ_YAML(bool, target_speed_curriculum, cfg["target_speed_curriculum"])
      READ_YAML(bool, quantizeVel, cfg["quantizeVel"])

      /// Reward coefficients
      READ_YAML(double, alive_bonus, cfg["alive_bonus"])
      READ_YAML(double, forwardVelRewardCoeff_, cfg["forwardVelRewardCoeff"])
      READ_YAML(double, lateralVelRewardCoeff_, cfg["lateralVelRewardCoeff"])
      READ_YAML(double, angularVelRewardCoeff_, cfg["angularVelRewardCoeff"])
      READ_YAML(double, torqueRewardCoeff_, cfg["torqueRewardCoeff"])
      READ_YAML(double, deltaTorqueRewardCoeff_, cfg["deltaTorqueRewardCoeff"])
      READ_YAML(double, actionRewardCoeff_, cfg["actionRewardCoeff"])
      READ_YAML(double, sidewaysRewardCoeff_, cfg["sidewaysRewardCoeff"])
      READ_YAML(double, jointSpeedRewardCoeff_, cfg["jointSpeedRewardCoeff"])
      READ_YAML(double, cost_coeff, cfg["cost_coeff"])
      READ_YAML(double, cost_decay_fac, cfg["cost_decay_fac"])
      READ_YAML(double, deltaContactRewardCoeff_, cfg["deltaContactRewardCoeff"])
      READ_YAML(double, deltaReleaseRewardCoeff_, cfg["deltaReleaseRewardCoeff"])
      READ_YAML(double, contactRewardCoeff_, cfg["contactRewardCoeff"])
      READ_YAML(double, footSlipRewardCoeff_, cfg["footSlipRewardCoeff"])
      READ_YAML(double, footClearenceRewardCoeff_, cfg["footClearenceRewardCoeff"])
      READ_YAML(double, contactChangeRewardCoeff_, cfg["contactChangeRewardCoeff"])
      READ_YAML(double, contactDistRewardCoeff_, cfg["contactDistRewardCoeff"])
      READ_YAML(double, upwardRewardCoeff_, cfg["upwardRewardCoeff"])
      READ_YAML(double, workRewardCoeff_, cfg["workRewardCoeff"])
      READ_YAML(double, yAccRewardCoeff_, cfg["yAccRewardCoeff"])
      READ_YAML(double, dynNoise_, cfg["dynNoise"])
      READ_YAML(double, max_speed, cfg["max_speed"])
      READ_YAML(double, lat_speed, cfg["lat_speed"])
      READ_YAML(double, ang_speed, cfg["ang_speed"])
      READ_YAML(double, terrain_freq, cfg["terrainFreq"])

      a1_ = world_->addArticulatedSystem(resourceDir_ + "/a1/urdf/a1.urdf");
      a1_->setName("a1");
      a1_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

      /// indices of links that should not make contact with ground
      footIndices_.insert(a1_->getBodyIdx("FR_calf"));
      footIndices_.insert(a1_->getBodyIdx("FL_calf"));
      footIndices_.insert(a1_->getBodyIdx("RR_calf"));
      footIndices_.insert(a1_->getBodyIdx("RL_calf"));

      base_idx = a1_->getBodyIdx("base");
      // head_idx =  a1_->getFrameIdxByName("head_joint");

      footVec_.push_back(a1_->getBodyIdx("FR_calf"));
      footVec_.push_back(a1_->getBodyIdx("FL_calf"));
      footVec_.push_back(a1_->getBodyIdx("RR_calf"));
      footVec_.push_back(a1_->getBodyIdx("RL_calf"));

      footFrame_.push_back(a1_->getFrameIdxByName("FR_foot_fixed"));
      footFrame_.push_back(a1_->getFrameIdxByName("FL_foot_fixed"));
      footFrame_.push_back(a1_->getFrameIdxByName("RR_foot_fixed"));
      footFrame_.push_back(a1_->getFrameIdxByName("RL_foot_fixed"));

      nFoot = footVec_.size();

      /// get robot data
      gcDim_ = a1_->getGeneralizedCoordinateDim();
      gvDim_ = a1_->getDOF();
      nJoints_ = gvDim_ - 6;
      base_mass_list = a1_->getMass();

      /// initialize containers
      gc_.setZero(gcDim_);
      gc_init_.setZero(gcDim_);
      gv_.setZero(gvDim_);
      gv_init_.setZero(gvDim_);
      pTarget_.setZero(gcDim_);
      vTarget_.setZero(gvDim_);
      pTarget12_.setZero(nJoints_);
      mass_noise_vector.setZero(base_mass_list.size());
      jt_mean_pos.setZero(nJoints_);

      jt_mean_pos << 0.05, 0.8, -1.4, -0.05, 0.8, -1.4, 0.05, 0.8, -1.4, -0.05, 0.8, -1.4;
      // randomize terrains
      randomize_terrain();

      /// set pd gains
      // std::cout << "Pid coeffs " <<  pid_coeff << std::endl;
      pgain = pid_coeff + Eigen::VectorXd::Random(1)[0] * 5.;
      dgain = 0.6 + Eigen::VectorXd::Random(1)[0] * 0.2;
      Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
      jointPgain.setZero();
      jointPgain.tail(nJoints_).setConstant(pgain);
      jointDgain.setZero();
      jointDgain.tail(nJoints_).setConstant(dgain);
      a1_->setPdGains(jointPgain, jointDgain);
      a1_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

      /// MUST BE DONE FOR ALL ENVIRONMENTS
      base_obDim_ = baseDim + (int)privinfo * (20 + 3 * (int)use_priv_vel + nFoot * (int)use_slope_dots + 1); // added step
      obDim_ = baseDim * history_len + base_obDim_;
      actionDim_ = nJoints_;
      actionMean_.setZero(actionDim_);
      actionStd_.setZero(actionDim_), invActionStd_.setZero(actionDim_);
      obDouble_.setZero(base_obDim_);
      ob_delay.setZero(base_obDim_);
      ob_concat.setZero(obDim_);

      /// action & observation scaling
      actionMean_ = gc_init_.tail(nJoints_);
      double act_std_val = 0.4;
      actionStd_.setConstant(act_std_val);
      invActionStd_.setConstant(1. / act_std_val);

      /// visualize if it is the first environment
      if (visualizable_)
      {
        server_ = std::make_unique<raisim::RaisimServer>(world_.get());
        server_->launchServer();
        server_->focusOn(a1_);

        if (isTest && use_slope_dots)
        { // TODO
          for (int visual_i = 0; visual_i < nFoot; visual_i++)
          {
            visual_scan_dots.push_back(server_->addVisualSphere("viz_sphere_" + std::to_string(visual_i),
                                                                0.1, 1, 0, 0, 1));
          }
        }
      }

      // to store history
      last_torque.resize(a1_->getGeneralizedForce().size());
      last_torque = VecDyn(a1_->getGeneralizedForce());
      last_contact.setZero(nFoot);
      grf_bin_obs.setZero(nFoot);
      last_swing.setZero(nFoot);
      last_foot_state.setZero(nFoot);
    }

    void init() final {}

    void kill_server()
    {
      server_->killServer();
    }

    void randomize_terrain()
    {
          add_random_terrain();
          gc_init_ << 0, 0, 0.6, 1.0, 0.0, 0.0, 0.0, jt_mean_pos;
    }

    void add_steps()
    {
      double sample_list[7] = {0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5};
      double stepSize = sample_list[std::rand() % 7];
      double stepHeight = 0.025 + 0.075 * Eigen::VectorXd::Random(1)[0];

      double pixelSize_ = 0.1;
      raisim::TerrainProperties terrainProp_;
      terrainProp_.xSize = 10.0;
      terrainProp_.ySize = 5.0;
      terrainProp_.xSamples = terrainProp_.xSize / pixelSize_;
      terrainProp_.ySamples = terrainProp_.ySize / pixelSize_;

      std::vector<double> heights_;
      heights_.resize(terrainProp_.xSamples * terrainProp_.ySamples);
      int xNum = terrainProp_.xSize / stepSize;
      int yNum = terrainProp_.ySize / stepSize;
      int gridWidth_ = stepSize / pixelSize_;

      Eigen::Map<Eigen::Matrix<double, -1, -1>> mapMat(heights_.data(),
                                                       terrainProp_.xSamples,
                                                       terrainProp_.ySamples);

      mapMat.setZero();
      /// steps
      for (size_t i = 0; i < xNum; i++)
      {
        for (size_t j = 0; j < yNum; j++)
        {
          double h = Eigen::VectorXd::Random(1)[0] * stepHeight;

          mapMat.block(gridWidth_ * i, gridWidth_ * j, gridWidth_, gridWidth_).setConstant(h);
        }
      }


      hm_ = world_->addHeightMap(terrainProp_.xSamples,
                                 terrainProp_.ySamples,
                                 terrainProp_.xSize,
                                 terrainProp_.ySize, 0.0, 0.0, heights_, "terrain");
    }

    void add_slope()
    {
      double stepHeight = 0.01 * Eigen::VectorXd::Random(1)[0];
      if (stepHeight >= 0)
        stepHeight += 0.005;
      else
        stepHeight -= 0.005;

      slope_info = stepHeight;

      double pixelSize_ = 0.05;
      raisim::TerrainProperties terrainProp_;
      terrainProp_.xSize = 15.0;
      terrainProp_.ySize = 15.0;
      terrainProp_.xSamples = terrainProp_.xSize / pixelSize_;
      terrainProp_.ySamples = terrainProp_.ySize / pixelSize_;

      std::vector<double> heights_;
      heights_.resize(terrainProp_.xSamples * terrainProp_.ySamples);

      double ht_i = -(double)terrainProp_.ySamples / 2.0 * stepHeight;
      for (int y = 0; y < terrainProp_.ySamples; y++)
      {
        for (int x = 0; x < terrainProp_.xSamples; x++)
        {
          size_t idx = y * terrainProp_.xSamples + x;
          if (x < 4 || x > terrainProp_.xSamples - 4 || y < 10 || y > terrainProp_.ySamples - 10)
            heights_[idx] = ht_i - 1.5;
          else
            heights_[idx] = ht_i + 0.02 * Eigen::VectorXd::Random(1)[0];
        }
        if (abs(y - (int)terrainProp_.ySamples / 2) > 10)
          ht_i += stepHeight;
      }

      Eigen::Map<Eigen::Matrix<double, -1, -1>> mapMat(heights_.data(),
                                                       terrainProp_.xSamples,
                                                       terrainProp_.ySamples);
      Eigen::Map<Eigen::Matrix<double, -1, 1>> mapVec(heights_.data(),
                                                      terrainProp_.xSamples * terrainProp_.ySamples,
                                                      1);
      Eigen::MatrixXd transMat = mapMat.transpose();
      Eigen::Map<Eigen::Matrix<double, -1, 1>> transVec(transMat.data(), terrainProp_.xSamples * terrainProp_.ySamples, 1);
      mapVec = transVec;

      hm_ = world_->addHeightMap(terrainProp_.ySamples,
                                 terrainProp_.xSamples,
                                 terrainProp_.ySize,
                                 terrainProp_.xSize, 0.0, 0.0, heights_, "terrain");
    }

    void add_single_step()
    {
      isSlope = true;
      int max_height_idx = std::max(2, (int)((itr_number - 3000) / 1000));
      max_height_idx = std::min(max_height_idx, (int)step_height_list.size());
      canonical_step_height = step_height_list[std::rand() % max_height_idx];

      double stepHeight = -canonical_step_height;
      double stepLength = 1.7; // should be enough to simulate multiple single steps

      double pixelSize_ = 0.02;
      double gridSize_ = 0.025;
      raisim::TerrainProperties terrainProp_;
      terrainProp_.xSize = 12.0;
      terrainProp_.ySize = 12.0;
      terrainProp_.xSamples = terrainProp_.xSize / pixelSize_;
      terrainProp_.ySamples = terrainProp_.ySize / pixelSize_;

      std::vector<double> heights_;
      heights_.resize(terrainProp_.xSamples * terrainProp_.ySamples);

      int N = (int)(stepLength / pixelSize_);
      int mid0 = 0.5 * terrainProp_.ySamples - (int)(0.5 / pixelSize_);
      int mid1 = 0.5 * terrainProp_.ySamples + (int)(0.7 / pixelSize_);
      double max = 0.3;
      double stepStart = max; // stepHeight; // * (mid0 / N);
      int cnt = 0;
      bool chamfer = false;
      for (int y = 0; y < mid0; y++)
      {
        if (cnt == N)
        {
          stepStart = max;
          cnt = 0;
          stepHeight *= -1;
        }
        if (cnt == 0 && Eigen::VectorXd::Random(1)[0] < 0.5)
          chamfer = true;
        else
          chamfer = false;
        for (int x = 0; x < terrainProp_.xSamples; x++)
        {
          size_t idx = y * terrainProp_.xSamples + x;
          max = stepStart + stepHeight;
          heights_[idx] = max;
          if (chamfer)
            heights_[idx] -= gridSize_;
        }
        cnt++;
      }

      for (int y = mid0; y < mid1; y++)
      {
        for (int x = 0; x < terrainProp_.xSamples; x++)
        {
          size_t idx = y * terrainProp_.xSamples + x;
          heights_[idx] = max;
        }
      }

      cnt = N;
      for (int y = mid1; y < terrainProp_.ySamples; y++)
      {
        if (cnt == N)
        {
          stepStart = max;
          cnt = 0;
          stepHeight *= -1;
        }
        if (cnt == 0 && Eigen::VectorXd::Random(1)[0] < 0.5)
          chamfer = true;
        else
          chamfer = false;
        for (int x = 0; x < terrainProp_.xSamples; x++)
        {
          size_t idx = y * terrainProp_.xSamples + x;
          max = stepStart + stepHeight;
          heights_[idx] = max;
          if (chamfer)
            heights_[idx] -= gridSize_;
        }
        cnt++;
      }

      Eigen::Map<Eigen::Matrix<double, -1, -1>> mapMat(heights_.data(),
                                                       terrainProp_.xSamples,
                                                       terrainProp_.ySamples);
      Eigen::Map<Eigen::Matrix<double, -1, 1>> mapVec(heights_.data(),
                                                      terrainProp_.xSamples * terrainProp_.ySamples,
                                                      1);
      Eigen::MatrixXd transMat = mapMat.transpose();
      Eigen::Map<Eigen::Matrix<double, -1, 1>> transVec(transMat.data(), terrainProp_.xSamples * terrainProp_.ySamples, 1);
      mapVec = transVec;

      hm_ = world_->addHeightMap(terrainProp_.ySamples,
                                 terrainProp_.xSamples,
                                 terrainProp_.ySize,
                                 terrainProp_.xSize, +terrainProp_.xSize / 3 + 1.0, 0.0, heights_, "terrain");
    }

    void add_stairs()
    {
      isSlope = true;
      canonical_step_height = step_height_list[std::rand() % max_height_idx];
      double sign = 1;
      bool down;
      down = Eigen::VectorXd::Random(1)[0] > 0;
      if (down){
        sign = -1.;
      	max_height_idx = std::min(max_height_idx, ((int)step_height_list.size()-3));
      	canonical_step_height = step_height_list[std::rand() % max_height_idx];
      }
      double stepHeight = sign * canonical_step_height; // 0.03 + sign * 0.05 * Eigen::VectorXd::Random(1)[0];

      // now step lenght
      double stepLength = step_length_list[std::rand() % max_step_idx];

      // fractal first
      raisim::TerrainProperties terrainProperties;
      terrainProperties.frequency = 20; // 10

      double zscale_val = 0.0;
      if (down)
	zscale_val = 0.20;
      if (isTest || isEval)
        zscale_val = 0.0;

      terrainProperties.zScale = zscale_val;
      terrainProperties.xSize = 12; // maybe 12
      terrainProperties.ySize = 12;
      terrainProperties.xSamples = 200; // maybe 200
      terrainProperties.ySamples = 200;
      terrainProperties.fractalOctaves = 2;
      terrainProperties.fractalLacunarity = 2.0;
      terrainProperties.fractalGain = 0.25;
      hm_ = world_->addHeightMap(0.0, 0.0, terrainProperties);

      double pixelSize_ = 0.02;
      double gridSize_ = 0.025;
      raisim::TerrainProperties terrainProp_;
      terrainProp_.xSize = 12.0;
      terrainProp_.ySize = 12.0;
      terrainProp_.xSamples = terrainProp_.xSize / pixelSize_;
      terrainProp_.ySamples = terrainProp_.ySize / pixelSize_;

      std::vector<double> heights_;
      heights_.resize(terrainProp_.xSamples * terrainProp_.ySamples);

      int N = (int)(stepLength / pixelSize_);
      int mids = (int)(1.2 / pixelSize_);
      int mid0 = 0.5 * terrainProp_.xSamples - (int)(1.2 / pixelSize_);
      int mid1 = 0.5 * terrainProp_.xSamples + (int)(0.0 / pixelSize_);
      double max = 0.2;
      double stepStart = max; // stepHeight; // * (mid0 / N);
      int cnt = 0;
      bool chamfer = false;
      // start platform
      for (int x = 0; x < mids; x++)
      {
        for (int y = 0; y < terrainProp_.xSamples; y++)
        {
          size_t idx = x * terrainProp_.xSamples + y;
          heights_[idx] = max;
        }
      }

      for (int x = mids; x < mid0; x++)
      {
        if (cnt == N)
        {
          stepStart = max;
          cnt = 0;
        }
        if (cnt == 0 && Eigen::VectorXd::Random(1)[0] < 0)
          chamfer = true;
        else
          chamfer = false;
        for (int y = 0; y < terrainProp_.xSamples; y++)
        {
          size_t idx = x * terrainProp_.xSamples + y;
          max = stepStart + stepHeight;
          heights_[idx] = max;
          if (chamfer)
            heights_[idx] -= gridSize_;
        }
        cnt++;
      }

      for (int x = mid0; x < mid1; x++)
      {
        for (int y = 0; y < terrainProp_.xSamples; y++)
        {
          size_t idx = x * terrainProp_.xSamples + y;
          heights_[idx] = max;
        }
      }

      cnt = N;
      for (int x = mid1; x < terrainProp_.ySamples; x++)
      {
        if (cnt == N)
        {
          stepStart = max;
          cnt = 0;
        }
        if (cnt == 0 && Eigen::VectorXd::Random(1)[0] < 0)
          chamfer = true;
        else
          chamfer = false;
        for (int y = 0; y < terrainProp_.xSamples; y++)
        {
          size_t idx = x * terrainProp_.xSamples + y;
          max = stepStart + stepHeight;
          heights_[idx] = max;
          if (chamfer)
            heights_[idx] -= gridSize_;
        }
        cnt++;
      }

      Eigen::Map<Eigen::Matrix<double, -1, -1>> mapMat(heights_.data(),
                                                       terrainProp_.xSamples,
                                                       terrainProp_.ySamples);
      Eigen::Map<Eigen::Matrix<double, -1, 1>> mapVec(heights_.data(),
                                                      terrainProp_.xSamples * terrainProp_.ySamples,
                                                      1);
      Eigen::MatrixXd transMat = mapMat.transpose();
      Eigen::Map<Eigen::Matrix<double, -1, 1>> transVec(transMat.data(), terrainProp_.xSamples * terrainProp_.ySamples, 1);
      mapVec = transVec;

      for (size_t i = 0; i < terrainProp_.xSamples; i++)
      {
        for (size_t j = 0; j < terrainProp_.ySamples; j++)
        {
          double xidx = i * pixelSize_ - terrainProp_.xSize / 2;
          double yidx = j * pixelSize_ - terrainProp_.ySize / 2;
          heights_[j * terrainProp_.xSamples + i] += hm_->getHeight(xidx, yidx);
        }
      }

      world_->removeObject(hm_);

      hm_ = world_->addHeightMap(terrainProp_.ySamples,
                                 terrainProp_.xSamples,
                                 terrainProp_.ySize,
                                 terrainProp_.xSize, terrainProp_.xSize / 2 - 0.6, 0.0, heights_, "terrain");
    }

    void randomize_sim_params()
    {
      if (randomize_friction)
      {
        double friction_list[6] = {0.5, 0.7, 0.9, 1.2, 1.5, 1.8};

        // ground friction
        friction = friction_list[std::rand() % 6]; // 1.0 + perturb_coeff * 0.9 * Eigen::VectorXd::Random(1)[0];
        world_->setDefaultMaterial(friction, 0.0, 0.0);
      }
      else
      {
        friction = 0.8;
        world_->setDefaultMaterial(friction, 0.0, 0.0);
      }

      if (randomize_gains)
      {
        // Resample Gains
        pgain = pid_coeff + Eigen::VectorXd::Random(1)[0] * 5.;
        dgain = 0.6 + Eigen::VectorXd::Random(1)[0] * 0.2;
        Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
        jointPgain.setZero();
        jointPgain.tail(nJoints_).setConstant(pgain);
        jointDgain.setZero();
        jointDgain.tail(nJoints_).setConstant(dgain);
        a1_->setPdGains(jointPgain, jointDgain);
      }
      else
      {
        pgain = pid_coeff;
        dgain = 0.6;
        Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
        jointPgain.setZero();
        jointPgain.tail(nJoints_).setConstant(pgain);
        jointDgain.setZero();
        jointDgain.tail(nJoints_).setConstant(dgain);
        a1_->setPdGains(jointPgain, jointDgain);
      }

      if (randomize_mass)
      {
        // updated mass variation
        auto mass_rand = 0.5 * (1. + Eigen::VectorXd::Random(1)[0]);
        a1_->getMass()[0] = 5.7 + 2.0 * mass_rand;
        Eigen::Vector3d com_vec = Eigen::VectorXd::Random(3);
        com_vec(0) = 0.1 * com_vec(0);
        com_vec(1) = 0.05 * com_vec(1);
        com_vec(2) = 0.0;
        a1_->getLinkCOM()[0] = com_vec;
        a1_->updateMassInfo();
        mass_params << mass_rand, com_vec(0), com_vec(1);
      }
      else
      {
        mass_params << 0.5, 0, 0;
      }

      if (randomize_motor_strength)
      {
        // resample motor strength
        motor_strength = Eigen::VectorXd::Ones(12) - 0.1 * Eigen::VectorXd::Random(12);
      }
      else
      {
        motor_strength = Eigen::VectorXd::Ones(12);
      }
    }

    void add_random_terrain()
    {
      isSlope = false;
      raisim::TerrainProperties terrainProperties;
      terrainProperties.frequency = 10; // 10

      double zscale_val = 0.22; // 0.22
      if (isTest)
        zscale_val = 0;

      terrainProperties.zScale = zscale_val;
      terrainProperties.xSize = 12; // maybe 12
      terrainProperties.ySize = 12;
      terrainProperties.xSamples = 200; // maybe 200
      terrainProperties.ySamples = 200;
      terrainProperties.fractalOctaves = 2;
      terrainProperties.fractalLacunarity = 2.0;
      terrainProperties.fractalGain = 0.25;
      hm_ = world_->addHeightMap(Eigen::VectorXd::Random(1)[0] + 4, Eigen::VectorXd::Random(1)[0], terrainProperties); // maybe revisit
      tparams[0] = 1;
      tparams[1] = 0.;
      tparams[2] = 0.;
      // tparams[rand_freq] = 1;
    }

    void sample_goals()
    {

      double delta_max_speed, delta_ang_speed;
      sample_goal_rand_num = Eigen::VectorXd::Random(1)[0];
      upwardRewardCoeff_ = 0.0;
      if (sample_goal_rand_num > 0.6)
      {                                                                            // 20%
        delta_max_speed = -0.5 + 0.15 * (Eigen::VectorXd::Random(1)[0] / 2 + 0.5); // 0. 
        delta_ang_speed = 0.0 * ((std::rand() % 2) * 2 - 1) * (0.6);               // 0. 
        actionRewardCoeff_ = -1.0;
        upwardRewardCoeff_ = -3.0;
        adaptiveForwardVelRewardCoeff_ = 2.0 * forwardVelRewardCoeff_;
        adaptiveAngularVelRewardCoeff_ = 2.0 * angularVelRewardCoeff_;
      }
      else
      {                                                                  // 90%
        delta_max_speed = -0.05 + 0.1 * Eigen::VectorXd::Random(1)[0]; // 0.35 ~ 0.55
        delta_ang_speed = 0.4 * Eigen::VectorXd::Random(1)[0];           // -0.4 ~ 0.4
        actionRewardCoeff_ = 0.;
        adaptiveForwardVelRewardCoeff_ = forwardVelRewardCoeff_;
        adaptiveAngularVelRewardCoeff_ = angularVelRewardCoeff_;
      }

      if (isSlope)
      {
        adaptiveForwardVelRewardCoeff_ = forwardVelRewardCoeff_;
        delta_max_speed = -0.05 + 0.05 * Eigen::VectorXd::Random(1)[0]; // 0.4 ~ 0.5
        delta_ang_speed = 0.2 * Eigen::VectorXd::Random(1)[0];          // -0.2 ~ 0.2
      }

      max_speed = 0.5 + delta_max_speed;
      ang_speed = 0.0 + delta_ang_speed;

      speed_vec.setZero();
      speed_vec << delta_max_speed, delta_ang_speed, delta_max_speed, delta_ang_speed;

      if (isTest)
      {
        std::cout << "speed_vec: " << speed_vec[0] + 0.5 << ", " << speed_vec[1] << std::endl;
      }
    }

    void compute_slope_info()
    {
      if (step_counter % 5 != 0)
        return;

      slope_dots.setZero(nFoot);
      if (not isSlope) return; // all zeros when on flat
      double yaw = bodyOrientation_[2];
      double cosy = std::cos(yaw);
      double siny = std::sin(yaw);

      double fx, fy, fz, bxi, byi, diff, tz;
      bool standing = true;

      std::vector<double> foot_height, foot_x, foot_y, look_px, look_py;

      for (int footIdx_i = 0; footIdx_i < nFoot; footIdx_i++)
      {
        raisim::Vec<3> footPosition;
        a1_->getFramePosition(footFrame_[footIdx_i], footPosition);

        fx = footPosition[0];
        fy = footPosition[1];
        foot_x.push_back(fx);
        foot_y.push_back(fy);
        fz = (hm_) ? hm_->getHeight(fx, fy) : 0;
        foot_height.push_back(fz);

        // points
        bxi = fx + lookahead_list[lookahead_idx] * cosy; //- dy * siny;
        byi = fy + lookahead_list[lookahead_idx] * siny; //+ dy * cosy;
        look_px.push_back(bxi);
        look_py.push_back(byi);
        tz = (hm_) ? hm_->getHeight(bxi, byi) : 0;

        if (footIdx_i < 2)
        {
          diff = tz - fz;
        }
        else
        {
          diff = fz - foot_height[footIdx_i - 2];
        }
        // clipping
        // std::cout << "Diff is " << diff << std::endl;
        if (abs(diff) < slope_th)
        {
          diff = 0;
        }
        else
        {
          standing = false;
        }
        slope_dots(footIdx_i) = diff;
        if (isTest && visualizable_)
        {
          if (slope_dots(footIdx_i) == 0)
            visual_scan_dots.at(footIdx_i)->color = {0, 1, 0, 1};
          else
            visual_scan_dots.at(footIdx_i)->color = {1, 0, 0, 1};
          visual_scan_dots.at(footIdx_i)->setPosition(bxi, byi, tz);
        }
      }
    }

    void reset(bool resample) final
    {
      // std::cout << "It number is " << itr_number << std::endl;
      if (isTest || isEval)
        resample = true;
      int change_terrain_freq = 100;
      if (itr_number > start_itr_stairs)
        change_terrain_freq = 50;
      if (itr_number % change_terrain_freq == 0 || isTest || isEval)
      {
        if (hm_)
        { // big change here, maybe revert
          world_->removeObject(hm_);
          int rand_terrain_select = 0;
          if (itr_number > start_itr_stairs || isTest || isEval)
          {
            rand_terrain_select = std::rand() % 100; // 2 + 1; // either step or terrains
          }

	     if (isEval) {
	       if (rand_terrain_select < 30) add_random_terrain();
	       else if (rand_terrain_select < 100) add_stairs();
	     } else {
	       if (rand_terrain_select < 40) add_random_terrain();
	       else if (rand_terrain_select < 100) add_stairs();
	     }
        }
        }

      randomize_sim_params();


      // whether to use or not future points
      ignore_future = false; // Eigen::VectorXd::Random(1)[0] > 0;

      avgXYVel.setZero();
      avgYawVel = 0;

      step_counter = 0;
      if (resample)
        sample_goals();
      gv_init_.setZero(gvDim_);
      gv_init_ += 0.1 * Eigen::VectorXd::Random(gvDim_);

      double z_ht = (hm_) ? hm_->getHeight(gc_init_[0], gc_init_[1]) : 0;
      gc_init_[2] = z_ht + 0.4;

      a1_->setState(gc_init_, gv_init_);

      if (isSlope and randomize_yaw)
      {
        double yaw = 0.5 * Eigen::VectorXd::Random(1)[0];
        Eigen::Vector3d eulerVec = {0.0, 0.0, yaw};
        Eigen::Matrix3d rot;
        eulerToRot(eulerVec, rot);
        a1_->setBaseOrientation_e(rot);
      }

      obs_history.clear();
      act_history.clear();
      jterr_history.clear();
      head_vel_history.clear();

      // get to rest pos
      pTarget_.tail(nJoints_) = Eigen::VectorXd::Zero(12) + actionMean_;

      a1_->setPdTarget(pTarget_, vTarget_);

      for (int i = 0; i < 50; i++)
      {
        if (server_)
          server_->lockVisualizationServerMutex();
        world_->integrate();
        if (server_)
          server_->unlockVisualizationServerMutex();
      }

      for (int j = 0; j < 50; j++)
        act_history.push_back(Eigen::VectorXd::Zero(12));
      for (int j = 0; j < 50; j++)
        jterr_history.push_back(Eigen::VectorXd::Zero(12));
      for (int j = 0; j < 50; j++)
        head_vel_history.push_back(Eigen::VectorXd::Zero(3));
      for (int j = 0; j < 100; j++)
        updateObservation();
      updateObservation();
      updateObservation();
      updateObservation();

      sample_residual_goal_steps = std::rand() % 50;
      sample_residual_env_steps = std::rand() % 10;
      // Update lookahead and stairs
    }

    void curriculumUpdate()
    {
      cost_coeff = 1.0; // pow(cost_coeff, cost_decay_fac); no decay in dagger
      // itr_number += 1; Do not increase complexity in dagger
    }

    void setSeed(int seed)
    {
      // seed_ = seed;
      std::srand(seed);
      for (int i = 0; i < 10; i++)
        clean_randomizer = Eigen::VectorXd::Random(1)[0];
    }

    void applyExternalForceRandomly()
    {
      // Vec<3> extForce = 150 * Eigen::VectorXd::Random(3);
      if (step_counter >= 200 && step_counter % 100 == 0)
      {
        Vec<3> extForce = 500 * Eigen::VectorXd::Random(3);
        for (int i = 0; i < 2; i++)
        {
          auto fi = extForce[i];
          extForce[i] = fi + (2.0 * (double)(fi > 0) - 1.0) * 400.0;
        }
        // std::cout << extForce[0] << "," <<extForce[1] << std::endl;
        extForce[2] = 0.0;
        a1_->setExternalForce(base_idx, extForce);
      }
    }

    double find_max(Eigen::VectorXd &swing_bin, Eigen::VectorXd &grf)
    {
      double max_val = 0.;
      for (int i = 0; i < nFoot; i++)
        if (swing_bin[i] == 0 and grf[i] > max_val)
          max_val = grf[i];
      return max_val;
    }

    double find_min(Eigen::VectorXd &swing_bin, Eigen::VectorXd &grf)
    {
      double min_val = 100.0;
      for (int i = 0; i < nFoot; i++)
        if (swing_bin[i] == 0 and grf[i] < min_val)
          min_val = grf[i];
      if (min_val == 100.0)
        return 0.;
      else
        return min_val;
    }


    double compute_forward_reward()
    {
      double r = 0.;
      double forward_r = adaptiveForwardVelRewardCoeff_ * (-std::abs(max_speed - bodyLinearVel_[0]) + std::abs(max_speed));
      double angular_r = adaptiveAngularVelRewardCoeff_ * (-std::abs(ang_speed - bodyAngularVel_[2]) + std::abs(ang_speed));
      r += forward_r;
      r += -abs(bodyLinearVel_[1]);
      r += angular_r;
      r += (alive_bonus * 1. / (2. - (double)isSlope));
      return r;
    }

    float step(const Eigen::Ref<EigenVec> &action_vec) final
    {
      step_counter += 1;

      if (step_counter % (300 + sample_residual_goal_steps) == 300 + sample_residual_goal_steps - 1)
        sample_goals();
      if (step_counter % (50 + sample_residual_env_steps) == 50 + sample_residual_env_steps - 1)
        randomize_sim_params();
      act_history.push_back(action_vec.cast<double>());

      auto action = act_history[act_history.size() - 3];

      // mean std normalize action
      pTarget12_ = action;
      pTarget12_ = pTarget12_.cwiseProduct(motor_strength); // adding motor strength perturbation
      pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
      pTarget12_ += actionMean_;
      pTarget_.tail(nJoints_) = pTarget12_;

      a1_->setPdTarget(pTarget_, vTarget_);

      Eigen::Vector2d loc_xy_prev = gc_.segment(0, 3);
      int rand_sim_steps = 0; // std::rand() % 3 - 1;
      for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1. * rand_sim_steps + 1e-10); i++)
      {
        if (server_)
          server_->lockVisualizationServerMutex();
        world_->integrate();
        if (server_)
          server_->unlockVisualizationServerMutex();
      }

      auto jterr = act_history[act_history.size() - 1] - (gc_.tail(12) - actionMean_).cwiseProduct(invActionStd_);
      jterr_history.push_back(jterr);

      // Foot contact events
      Eigen::VectorXd grf;
      Eigen::VectorXd grf_bin;
      Eigen::VectorXd swing_bin;
      Eigen::VectorXd foot_vel;
      Eigen::VectorXd foot_pos_ht;
      Eigen::VectorXd foot_pos_bin;
      Eigen::VectorXd foot_pos_err;
      Eigen::VectorXd current_foot_state;
      raisim::Vec<3> footVelocity;
      raisim::Vec<3> footPosition;
      // raisim::Vec<3> net_impulse;
      double clearence = 0.04;
      grf.setZero(nFoot);
      grf_bin.setZero(nFoot);
      swing_bin.setOnes(nFoot);
      foot_vel.setZero(nFoot);
      foot_pos_ht.setZero(nFoot);
      foot_pos_bin.setZero(nFoot);
      foot_pos_err.setZero(nFoot);
      current_foot_state = last_foot_state;
      for (int footIdx_i = 0; footIdx_i < nFoot; footIdx_i++)
      {
        auto footIndex = footVec_[footIdx_i];
        // check for contact event
        for (auto &contact : a1_->getContacts())
        {
          if (contact.skip())
            continue;
          if (footIndex == contact.getlocalBodyIndex())
          {
            auto impulse_i = (contact.getContactFrame().e() * contact.getImpulse()->e()).norm();
            if (impulse_i > 0)
            {
              grf[footIdx_i] += impulse_i;
              grf_bin[footIdx_i] = 1.0;
              swing_bin[footIdx_i] = 0.0;
              // net_impulse += impulse_i;
            }
          }
        }
        // measure foot velocity
        a1_->getFrameVelocity(footFrame_[footIdx_i], footVelocity);
        a1_->getFramePosition(footFrame_[footIdx_i], footPosition);
        foot_vel[footIdx_i] = footVelocity.squaredNorm();
        foot_pos_bin[footIdx_i] = (double)(footPosition[2] > clearence);
        foot_pos_ht[footIdx_i] = footPosition[2];
        foot_pos_err[footIdx_i] = pow(footPosition[2] - clearence, 2);
      }
      grf_bin_obs = grf_bin;

      if(abs(max_speed) < 0.15)
      	applyExternalForceRandomly();
      updateObservation();

      // recording relevant values
      VecDyn current_torque = a1_->getGeneralizedForce();
      current_torque_squareNorm = current_torque.squaredNorm();

      // count contact change
      for (int fi = 0; fi < nFoot; fi++)
      {
        if (foot_pos_bin[fi] == 1)
          current_foot_state[fi] = 1.0;
        else if (swing_bin[fi] == 0)
          current_foot_state[fi] = -1.0;
      }
      double contact_changes = 0.0;
      for (int fi = 0; fi < nFoot; fi++)
        if ((current_foot_state[fi] + last_foot_state[fi]) == 0.0)
          contact_changes += 1.0;

      // reward compuation
      forwardReward = compute_forward_reward();

      deltaContactReward_ = ((grf - last_contact).array().max(0).square().sum());
      deltaReleaseReward_ = ((grf - last_contact).array().min(0).square().sum());
      contactReward_ = grf.squaredNorm();

      contactDistReward_ = pow((find_max(swing_bin, grf) - find_min(swing_bin, grf)), 2);

      // removing torque for sideways motion
      torqueReward_ = (current_torque.squaredNorm() - pow(current_torque[6], 2) - pow(current_torque[9], 2) - pow(current_torque[12], 2) - pow(current_torque[15], 2));

      deltaTorqueReward_ = (current_torque.e() - last_torque.e()).squaredNorm();
      actionReward_ = action.squaredNorm();
      sidewaysReward_ = (pow(action[0], 2) + pow(action[3], 2) + pow(action[6], 2) + pow(action[9], 2));
      jointSpeedReward_ = a1_->getGeneralizedVelocity().e().tail(nJoints_).squaredNorm();
      footSlipReward_ = (foot_vel.transpose() * grf_bin).sum();
      footClearenceReward_ = (swing_bin.transpose() * foot_pos_err).sum();
      contactChangeReward_ = contact_changes;
      upwardReward_ = bodyOrientation_.head(2).squaredNorm();
      workReward_ = (a1_->getGeneralizedVelocity().e().tail(nJoints_).transpose() * current_torque.e().tail(nJoints_)).sum();
      yAccReward_ = pow(bodyLinearVel_[2], 2);

      // update histories
      last_torque = VecDyn(a1_->getGeneralizedForce());
      last_contact = grf;
      last_swing = swing_bin;
      last_foot_state = current_foot_state;

      // double targetSpeedRewardScale = 1.0; (forwardVelRewardCoeff_ / ((target_speed / 0.375 - 1) / targetSpeedRewardScale + 1))
      auto cumulative_reward = forwardReward + workReward_ * workRewardCoeff_ +
                               cost_coeff * (footSlipReward_ * footSlipRewardCoeff_ + torqueReward_ * torqueRewardCoeff_ + contactReward_ * contactRewardCoeff_ + deltaContactReward_ * deltaContactRewardCoeff_ + deltaReleaseReward_ * deltaReleaseRewardCoeff_ +
                                             deltaTorqueReward_ * deltaTorqueRewardCoeff_ + actionReward_ * actionRewardCoeff_ + jointSpeedReward_ * jointSpeedRewardCoeff_ +
                                             footClearenceReward_ * footClearenceRewardCoeff_ + upwardReward_ * upwardRewardCoeff_ +
                                             yAccReward_ * yAccRewardCoeff_ + contactDistReward_ * contactDistRewardCoeff_ +
                                             contactChangeReward_ * contactChangeRewardCoeff_ + sidewaysReward_ * sidewaysRewardCoeff_);

      cumulative_reward /= 100;

      // sometimes zero velocity leads to unstable training. So explicitly checking for nans and setting the velocities to zero
      if (isnan(cumulative_reward))
      {
        cumulative_reward = -50;
        bodyLinearVel_.setZero(3);
        gv_.setZero(gvDim_);
      }

      if (cumulative_reward > 100.0 || cumulative_reward < -100.0)
        cumulative_reward = -10.0;

      walked_dist_ = (gc_.segment(0, 3) - loc_xy_prev).norm();
      return cumulative_reward;
    }

    void updateObservation()
    {
      a1_->getState(gc_, gv_);
      raisim::Vec<4> quat;
      raisim::Mat<3, 3> rot;
      quat[0] = gc_[3];
      quat[1] = gc_[4];
      quat[2] = gc_[5];
      quat[3] = gc_[6];
      // Eigen::Quaternionf e_quat(quat[0], quat[1], quat[2], quat[3]);//(quat);
      raisim::quatToRotMat(quat, rot);
      bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
      bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);

      if (step_counter <= 1)
      {
        avgXYVel = bodyLinearVel_.segment(0, 2);
        avgYawVel = bodyAngularVel_[2];
      }
      else
      {
        avgXYVel = (1 - bodyLinearVel_avg_weight) * avgXYVel + bodyLinearVel_avg_weight * bodyLinearVel_.segment(0, 2);
        avgYawVel = (1 - bodyLinearVel_avg_weight) * avgYawVel + bodyLinearVel_avg_weight * bodyAngularVel_[2];
      }

      if (isTest && step_counter % 50 == 1)
        std::cout << "\t actual speed: " << bodyLinearVel_[0] << ' ' << bodyLinearVel_[1]  << ' ' << bodyAngularVel_[2] << std::endl;

      quatToEuler(quat, bodyOrientation_);
      double phase = (step_counter % 100) * 1.0 / 50.0;

      if (use_slope_dots)
        compute_slope_info();

      obDouble_ << bodyOrientation_.head(2),
          gc_.tail(12), /// joint angles
          gv_.tail(12),
          act_history[act_history.size() - 1],
          speed_vec,
          (double)includeGRF * grf_bin_obs,
          mass_params,
	  motor_strength,
          friction,
          avgXYVel, avgYawVel,
          slope_dots,
          isSlope;

      obs_history.push_back(obDouble_);
    }

    void observe(Eigen::Ref<EigenVec> ob) final
    {
      int lag = 1;
      int vec_size = obs_history.size();
      ob_delay << obs_history[vec_size - lag]; //, obs_history[obs_history.size() - 2], obs_history[obs_history.size() - 3];
      for (int i = 0; i < history_len; i++)
      {
        ob_concat.segment(baseDim * i, baseDim) << obs_history[vec_size - lag - history_len + i].head(baseDim);
      }
      ob_concat.tail(base_obDim_) << ob_delay;
      ob = ob_concat.cast<float>();
    }

    bool isTerminalState(float &terminalReward) final
    {
      terminalReward = float(terminalRewardCoeff_);

      // if the contact body is not feet
      if (itr_number < 100000)
      {
        for (auto &contact : a1_->getContacts())
          if (footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end())
          {
            // std::cout << "Terminating for contact " << std::endl;
            return true;
          }
      }

      float term_pitch = 0.2;
      if ((isTest || isSlope || isEval))
        term_pitch = 0.8;
      if (abs(bodyOrientation_[0]) > 0.6 || abs(bodyOrientation_[1]) > term_pitch)
      {
        //std::cout << "Terminating for pitch " << std::endl;
        return true;
      }

      double x = gc_[0];
      double y = gc_[1];
      double z_ht = (hm_) ? hm_->getHeight(x, y) : 0;
      double term_height = 0.24;
      if ((isTest || isSlope || isEval))
        term_height = 0.1;
      if ((gc_[2] - z_ht) < term_height)
      {
        //std::cout << "Terminating for height " << std::endl;
        return true;
      }

      if (isSlope)
      {
        double yaw = bodyOrientation_[2];
        if (abs(yaw) > 0.8) {
          //std::cout << "Terminating for yaw " << std::endl;
          return true;
        }
      }

      terminalReward = 0.f;
      int max_steps = 1200;
      if (isSlope)
        max_steps = 1200;

      if (step_counter > max_steps)
        return true;

      return false;
    }

    void getDis(Eigen::Ref<EigenVec> dis)
    {
      return;
    }

    void getRewardInfo(Eigen::Ref<EigenVec> reward)
    {
      reward << forwardReward, 0, 0, deltaTorqueReward_, actionReward_, sidewaysReward_, jointSpeedReward_, deltaContactReward_, deltaReleaseReward_, footSlipReward_, upwardReward_, workReward_, yAccReward_, current_torque_squareNorm, canonical_step_height, walked_dist_;
      return;
    }

  private:
    int gcDim_, gvDim_, nJoints_;
    bool visualizable_ = false;
    raisim::ArticulatedSystem *a1_;
    Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
    double terminalRewardCoeff_ = -10.;
    double forwardVelRewardCoeff_ = 0., forwardVelReward_ = 0.;
    double lateralVelRewardCoeff_ = 0., lateralVelReward_ = 0.;
    double angularVelRewardCoeff_ = 0., angularVelReward_ = 0.;
    double torqueRewardCoeff_ = 0., torqueReward_ = 0.;
    double deltaTorqueRewardCoeff_ = 0., deltaTorqueReward_ = 0.;
    double jointSpeedRewardCoeff_ = 0., jointSpeedReward_ = 0.;
    double actionRewardCoeff_ = 0., actionReward_ = 0.;
    double sidewaysRewardCoeff_ = 0., sidewaysReward_ = 0.;
    double deltaContactRewardCoeff_ = 0., deltaContactReward_ = 0.;
    double deltaReleaseRewardCoeff_ = 0., deltaReleaseReward_ = 0.;
    double contactRewardCoeff_ = 0., contactReward_ = 0.;
    double footSlipRewardCoeff_ = 0., footSlipReward_ = 0.;
    double footClearenceRewardCoeff_ = 0., footClearenceReward_ = 0.;
    double contactChangeRewardCoeff_ = 0., contactChangeReward_ = 0.;
    double contactDistRewardCoeff_ = 0., contactDistReward_ = 0.;
    double upwardRewardCoeff_ = 0., upwardReward_ = 0.;
    double workRewardCoeff_ = 0., workReward_ = 0.;
    double yAccRewardCoeff_ = 0., yAccReward_ = 0.;
    double cost_coeff = 0., cost_decay_fac = 0.;
    double friction = 0.8;
    VecDyn last_torque;
    Eigen::VectorXd actionMean_, invActionStd_, actionStd_, obDouble_, ob_delay;
    Eigen::Vector3d bodyLinearVel_, bodyAngularVel_, bodyOrientation_;
    std::set<size_t> footIndices_;
    std::vector<size_t> footVec_;
    std::vector<size_t> footFrame_;
    Eigen::VectorXd last_contact;
    Eigen::VectorXd grf_bin_obs;
    Eigen::VectorXd last_swing;
    Eigen::Vector3d mass_params;
    Eigen::VectorXd last_foot_state;
    Eigen::VectorXd motor_strength;
    Eigen::VectorXd mass_noise_vector;
    Eigen::VectorXd jt_mean_pos;
    double max_speed = 0.0;
    double pgain = 50.;
    double dgain = 0.6;
    double lat_speed = 0.0;
    double ang_speed = 0.0;
    double slope_info = 0.0;
    Eigen::VectorXd slope_dots;
    bool roughTerrain, includeGRF;
    bool privinfo;
    double dynNoise_ = 0.0;
    int nFoot = 0;
    size_t base_idx;
    Eigen::Vector3d des_speed_vector;
    Eigen::Vector4d speed_vec;
    double tparams[3] = {0., 0., 0.};
    int step_counter = 0;
    double terrain_freq = 0;
    double canonical_step_height = 0.1; // starting point
    bool isSlope = false;
    bool quantizeVel = false;
    raisim::HeightMap *hm_ = nullptr;
    std::vector<int> link_ids;
    std::vector<double> base_mass_list;
    std::vector<double> step_height_list = {0.1, 0.13, 0.15, 0.17, 0.19, 0.21};
    int max_height_idx = (int) step_height_list.size();
    int start_itr_stairs = 1000;
    std::deque<Eigen::VectorXd> obs_history;
    std::deque<Eigen::VectorXd> act_history;
    std::deque<Eigen::VectorXd> jterr_history;
    std::deque<Eigen::VectorXd> head_vel_history;
    bool isTest;
    std::vector<double> lookahead_list = {0.05};
    int lookahead_idx = 0;
    std::vector<double> step_length_list = {0.6, 0.5, 0.4, 0.3};
    int max_step_idx = (int)step_length_list.size();

    bool randomize_friction;
    bool randomize_mass;
    bool randomize_motor_strength;
    bool randomize_gains;

    bool use_priv_vel = true;
    bool use_slope_dots = false;

    bool cts_target_speed;
    double target_speed;
    double target_speed_period;
    double target_end_speed;
    double target_start_speed;
    bool speedTest;
    bool target_speed_curriculum;

    int baseDim, geomDim;
    Eigen::Vector2d avgXYVel;
    double avgYawVel;
    double bodyLinearVel_avg_weight;

    double forwardRewardX = 0;
    double forwardRewardY = 0;
    double forwardRewardZ = 0;
    double forwardReward;
    double pid_coeff = 55;
    double slope_th = 0.5;

    double current_torque_squareNorm;

    int env_id_, n_futures;
    double clean_randomizer;

    bool isEval;

    int sample_residual_goal_steps;
    int sample_residual_env_steps;
    bool randomize_yaw = false;

    int base_obDim_;
    Eigen::VectorXd ob_concat;
    int history_len;

    double adaptiveAngularVelRewardCoeff_ = 0.;
    double adaptiveForwardVelRewardCoeff_ = 0.;
    double adaptiveActionRewardCoeff_ = 0.;
    double adaptiveSidewaysRewardCoeff_ = 0.;
    double walked_dist_ = 0.;
    bool ignore_future = true;

    double sample_goal_rand_num = 0.0;
    double alive_bonus = 0.0;

    size_t head_idx;
    raisim::Vec<3> gc_headVelocity, gc_headAngularVelocity;
    raisim::Mat<3, 3> gc_headOrientation;
    Eigen::Vector3d headLinearVel_, headAngularVel_;
    std::vector<raisim::Visuals *> visual_scan_dots;
  };
}
