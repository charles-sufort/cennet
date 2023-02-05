#include <iostream>
#include <string>
#include <map>
#include <Eigen/Core>
#include <math.h>
#include "rnn_f.h"

using namespace std;
using namespace Eigen;

RNN_f::RNN_f(int P_d_t, int P_d_x, int P_d_y, int P_d_h, int P_max_iters, double P_epoch, double P_tol):RNN(P_max_iters, P_epoch, P_tol,P_d_t,P_d_y){
	d_t = P_d_t;
	d_x = P_d_x;
	d_y = P_d_y;
	d_h = P_d_h;
	W = MatrixXd::Random(d_h,d_x);
	U = MatrixXd::Random(d_h,d_h);
	V = MatrixXd::Random(d_y,d_h);
	A = MatrixXd::Zero(d_h,d_t);
	O = MatrixXd::Zero(d_y,d_t);
	H = MatrixXd::Zero(d_h,d_t+1);
	b = VectorXd::Random(d_h);
	c = VectorXd::Random(d_y);
	Y_bar = MatrixXd::Zero(d_y,d_t);
	Y = MatrixXd::Zero(d_y,d_t);
	X = MatrixXd::Zero(d_x,d_t);
	del_W = MatrixXd::Zero(d_h,d_x);
	del_U = MatrixXd::Zero(d_h,d_h);
	del_V = MatrixXd::Zero(d_y,d_h);
	del_A = MatrixXd::Zero(d_h,d_t);
	del_O = MatrixXd::Zero(d_y,d_t);
	del_H = MatrixXd::Zero(d_h,d_t);
	del_b = VectorXd::Zero(d_h);
	del_c = VectorXd::Zero(d_y);
	acc_W = MatrixXd::Zero(d_h,d_x);
	acc_U = MatrixXd::Zero(d_h,d_h);
	acc_V = MatrixXd::Zero(d_y,d_h);
	acc_b = VectorXd::Zero(d_h);
	acc_c = VectorXd::Zero(d_y);
}

void RNN_f::forward_propagation(MatrixXd X_P){
	VectorXd a = VectorXd::Zero(d_h);
	X = X_P;
	for(int t=0; t<d_t; t++){
		a = U*H.col(t) + W*X.col(t) + b;
		for(int j=0;j<d_h;j++){
			H.col(t+1)[j] = tanh(a[j]);
		}
		O.col(t) = V*H.col(t+1) + c;
		Y_bar.col(t) = softmax(O.col(t));
	}
}

void RNN_f::backward_propagation(MatrixXd X_P, MatrixXd Y_P){
	Y = Y_P;
	forward_propagation(X_P);
	delta_o();
	delta_h();
	delta_u();
	delta_c();
	delta_w();
	delta_b();
}

void RNN_f::update_model(int method){
	MatrixXd *M_arr[3] = {&V,&W,&U};	
	MatrixXd *del_M_arr[3] = {&del_V,&del_W,&del_U};
	VectorXd *V_arr[2] = {&b, &c};
	VectorXd *del_V_arr[2] = {&del_b, &del_c};
	if (method == 1){
		update_gd(M_arr,del_M_arr, V_arr, del_V_arr,3, 2);
	}
}

void RNN_f::grad_acc(){
	acc_W = acc_W + del_W;
	acc_U = acc_U + del_U;
	acc_V = acc_V + del_V;
	acc_b = acc_b + del_b;
	acc_c = acc_c + del_c;

}

void RNN_f::reset_acc(){
	acc_W = MatrixXd::Zero(d_h,d_x);
	acc_U = MatrixXd::Zero(d_h,d_h);
	acc_V = MatrixXd::Zero(d_y,d_h);
	acc_b = VectorXd::Zero(d_h);
	acc_c = VectorXd::Zero(d_y);    	
}

void RNN_f::delta_o(){
        //[del_o]_i = y_bar_i^(t) - \delta_{ik}
	del_O = Y_bar - Y;
}

// delta_h^tau L =  V^T(\delta_o^tau)
// delta_h^t L = W^Tdiag(1-(h^{t+1})^2)(delta_h^{t+1} L) + V^T(\delta_o^t)
void RNN_f::delta_h(){
	del_H.col(d_t-1) = (V.transpose())*del_O.col(d_t-1);
	for(int t=d_t-2; t<-1; t--){
		del_H.col(t) = (W.transpose())*((MatrixXd )(VectorXd::Ones(d_h) -((VectorXd) H.col(t+2).array().square() )).asDiagonal())*del_H.col(t+1);
		del_H.col(t) = del_H.col(t) + (V.transpose())*del_O.col(t);
	}
}

// delta_V L = \sum_t delta_o^(t) h^t
void RNN_f::delta_v(){
	del_V = MatrixXd::Zero(d_y,d_h);
	for (int t=0; t<d_t; t++){
		del_V = del_V + del_O.col(t)*(H.col(t).transpose());
	}
}

// delta_c L = sum_t delta_o
void RNN_f::delta_c(){
	del_c = VectorXd::Zero(d_y);
	for (int t=0; t<d_t; t++){
		del_c = del_c + del_O.col(t);
	}
}

// delta_W L = sum_t diag(1-(h^(t))^2)(delta_h^(t) L)(x^(t))^T
void RNN_f::delta_w(){
	del_W = MatrixXd::Zero(d_h,d_x);
	for (int t=0; t<d_t; t++){
		MatrixXd HXT = del_H.col(t)*(X.col(t).transpose());
		VectorXd o = VectorXd::Ones(d_h);
		VectorXd s = H.col(t+1).array().square();
		MatrixXd N = (o-s).asDiagonal(); 
		del_W = del_W + N*HXT;
	}
}

// delta_U L = sum_t diag(1-(h^(t))^2)(delta_h^(t) L)(h^(t-1))^T
void RNN_f::delta_u(){
	del_U = MatrixXd::Zero(d_h,d_h);
	for (int t=0; t<d_t; t++){
		MatrixXd DHT = del_H.col(t)*(H.col(t).transpose());
		VectorXd o = VectorXd::Ones(d_h);
		VectorXd s = H.col(t+1).array().square();
		MatrixXd N = (o-s).asDiagonal(); 
		del_U = del_U + N*DHT;
	}
}

//  delta_b L = sum_t diag(1-(h^(t))^2)delta_h^(t) L
void RNN_f::delta_b(){
	del_b = VectorXd::Zero(d_h);
	for (int t=0; t<d_t; t++){
		del_b = ((MatrixXd )(VectorXd::Ones(d_h) -((VectorXd) H.col(t+1).array().square() )).asDiagonal())*del_H.col(t);
	}
}

void RNN_f::get_Y_pred(int **y_p_p){
	for(int t=0;t<d_t;t++){
		(*y_p_p[t]) = 0;
	}
	for(int t=0;t<d_t;t++){
		for(int j=0;j<d_y; j++){
			if (Y_bar(j,t) > (*y_p_p[t])){
				*y_p_p[t] = Y_bar(j,t);
			}
		}
	}
}

