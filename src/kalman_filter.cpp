#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

#define PI 3.14159265358979323846

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
    * predict the state
  */
  	x_ = F_ * x_;
	MatrixXd Ft = F_.transpose();
	P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
    * update the state by using Kalman Filter equations
  */
  	VectorXd z_pred = H_ * x_;
	VectorXd y = z - z_pred;
        commonKF(z, y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
    * update the state by using Extended Kalman Filter equations
  */
  float x_pos = x_(0);
  float y_pos = x_(1);
  float vx = x_(2);
  float vy = x_(3);
  
  float rho = sqrt(x_pos*x_pos+y_pos*y_pos);
  float phi = atan2(y_pos,x_pos);
  float ro_dot = (x_pos*vx+y_pos*vy)/rho;
  VectorXd z_pred = VectorXd(3);
  z_pred << rho, phi, ro_dot;
  
  VectorXd y = z - z_pred;
  y(1) = normalizeAngle(y(1));
  commonKF(z, y);
}

float KalmanFilter::normalizeAngle(float phi)
{
  if (phi > PI)
  {
    do 
    {
      phi -= 2*PI;
    } while(phi > PI);
  }
  if (phi < -PI)
  {
    do 
    {
      phi += 2*PI;
    } while(phi < -PI);
  }
  return phi; 
}


void KalmanFilter::commonKF(const Eigen::VectorXd &z, const Eigen::VectorXd &y)
{
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;
  
  x_ = x_ + (K*y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
