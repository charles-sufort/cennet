#include <iostream>
#include <string>
#include <map>
#include <Eigen/Core>
#include "rnn.h"

using namespace std;
using namespace Eigen;

RNN::RNN(int P_max_iters, float P_epoch, float P_tol){
	max_iters = P_max_iters;
	epoch = P_epoch;
	tol = P_tol;
}

void RNN::train(MatrixXf X_S[], MatrixXf Y_S[],int method, struct cfg meth_cfg ){
	cout << meth_cfg.int_opts.find("n_batch")->second << endl;
	cout << X_S[0] << endl;
	cout << Y_S[0] << endl;
}

VectorXf RNN::softmax(VectorXf x){
	VectorXf x_exp = x.array().exp();
	float s = x_exp.array().sum();
	x_exp = (1/s)*x_exp.array();
	return x_exp;
}

void RNN::batch_gradient(MatrixXf X_S[], MatrixXf Y_S[], int method, struct cfg batch_cfg){
	
}

void RNN::update1(MatrixXf **M_arr,MatrixXf **del_M_arr,VectorXf **V_arr,VectorXf **del_V_arr,int m_n, int v_n){
	for(int i=0;i<m_n;i++){
		*(M_arr[i]) = *(M_arr[i]) - (MatrixXf) (epoch*((*del_M_arr[i]).array()));
	}
	for(int i=0;i<v_n;i++){
		*(V_arr[i]) = *(V_arr[i]) - (VectorXf) (epoch*((*del_V_arr[i]).array()));
	}
}
