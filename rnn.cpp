#include <iostream>
#include <string>
#include <map>
#include <Eigen/Core>
#include "rnn.h"

using namespace std;
using namespace Eigen;

RNN::RNN(int P_max_iters, double P_epoch, double P_tol, int P_d_t, int P_d_y){
	d_t = P_d_t;
	d_y = P_d_y;
	max_iters = P_max_iters;
	epoch = P_epoch;
	tol = P_tol;
}

void RNN::train(MatrixXd X_S[], MatrixXd Y_S[],int method, struct cfg meth_cfg, int n){
	std::vector<int> range;
}

double RNN::test(MatrixXd X_S[], MatrixXd Y_S[], int n){
	int n_c = 0;
	int y_p[d_t];
	int *y_p_p[d_t];
	for(int t=0;t<d_t;t++){
		y_p_p[t] = &y_p[t];
	}
	for (int i=0;i<n; i++){
		forward_propagation(X_S[i]);
		get_Y_pred(y_p_p);
		cout << y_p << endl;
		for (int t=0;t<d_t;t++){
			if (Y_S[i](y_p[t],t) == 1){
				n_c++;
			}
		}
	}
	return n_c;
}

VectorXd RNN::softmax(VectorXd x){
	VectorXd x_exp = x.array().exp();
	double s = x_exp.array().sum();
	x_exp = (1/s)*x_exp.array();
	return x_exp;
}

void RNN::batch_gradient(MatrixXd X_S[], MatrixXd Y_S[], struct cfg batch_cfg, int n){
	int n_batch = batch_cfg.int_opts.find("n_batch")->second;
	int err[max_iters];
	int i_b;
	int iter = 0;
	bool converged = false;
	std::vector<int> range;
	for (int i=0; i<n_batch; i++){
		range.push_back(i);
		err[i] = 0;
	}
	while (!converged & iter < max_iters){
		random_shuffle(range.begin(), range.end());
		for(int i=0; i<n_batch; i++){
			MatrixXd X = X_S[range[i]];
			MatrixXd Y = Y_S[range[i]];
			backward_propagation(X,Y);
			grad_acc();
		}
		update_model();
		reset_acc();
		iter++;
	}
	
}


void RNN::update_gd(MatrixXd **M_arr,MatrixXd **del_M_arr,VectorXd **V_arr,VectorXd **del_V_arr,int m_n, int v_n){
	for(int i=0;i<m_n;i++){
		*(M_arr[i]) = *(M_arr[i]) - (MatrixXd) (epoch*((*del_M_arr[i]).array()));
	}
	for(int i=0;i<v_n;i++){
		*(V_arr[i]) = *(V_arr[i]) - (VectorXd) (epoch*((*del_V_arr[i]).array()));
	}
}

void RNN::grad_acc(){}
void RNN::reset_acc(){}
void RNN::update_model(){}

void RNN::forward_propagation(MatrixXd X_P){
}

void RNN::backward_propagation(MatrixXd X_P, MatrixXd Y_P){
}

void RNN::get_Y_pred(int** y_p_p){}

