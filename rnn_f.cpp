#include <iostream>
#include <string>
#include <map>
#include <Eigen/Core>
#include "rnn_f.h"

using namespace std;
using namespace Eigen;

RNN_f::RNN_f(int r_d_t, int r_d_x, int r_d_y, int r_d_h, int P_max_iters, float P_epoch, float P_tol):RNN(P_max_iters, P_epoch, P_tol){
	d_t = r_d_t;
	d_x = r_d_x;
	d_y = r_d_y;
	d_h = r_d_h;
	W = MatrixXf::Random(d_h,d_h);
	U = MatrixXf::Random(d_h,d_x);
	V = MatrixXf::Random(d_y,d_h);
	A = MatrixXf::Zero(d_h,d_t);
	O = MatrixXf::Zero(d_y,d_t);
	H = MatrixXf::Zero(d_h,d_t+1);
	b = VectorXf::Random(d_h);
	c = VectorXf::Random(d_y);
	Y_bar = MatrixXf::Zero(d_y,d_t);
	Y = MatrixXf::Zero(d_y,d_t);
	X = MatrixXf::Zero(d_x,d_t);
	del_W = MatrixXf::Zero(d_h,d_h);
	del_U = MatrixXf::Zero(d_h,d_x);
	del_V = MatrixXf::Zero(d_y,d_h);
	del_A = MatrixXf::Zero(d_h,d_t);
	del_O = MatrixXf::Zero(d_y,d_t);
	del_H = MatrixXf::Zero(d_h,d_t);
	del_b = VectorXf::Zero(d_h);
	del_c = VectorXf::Zero(d_y);
	acc_W = MatrixXf::Zero(d_h,d_h);
	acc_U = MatrixXf::Zero(d_h,d_x);
	acc_V = MatrixXf::Zero(d_y,d_h);
	acc_b = VectorXf::Zero(d_h);
	acc_c = VectorXf::Zero(d_y);
}

void RNN_f::forward_propagation(MatrixXf X_P){
	VectorXf a = VectorXf::Zero(d_h);
	X = X_P;
	for(int t=0; t<d_t; t++){
		a = U*H.col(t) + W*X.col(t) + b;
		H.col(t+1) = a.array().atanh();
		O.col(t) = V*H.col(t+1) + c;
		Y_bar.col(t) = softmax(O.col(t));
	}
}

void RNN_f::backward_propagation(MatrixXf X_P, MatrixXf Y_P){
	Y = Y_P;
	forward_propagation(X_P);
	cout << "pre o" << endl;
	delta_o();
	cout << "pre h" << endl;
	delta_h();
	cout << "pre u" << endl;
	delta_u();
	cout << "pre c" << endl;
	delta_c();
	cout << "pre w" << endl;
	delta_w();
	cout << "pre b" << endl;
	delta_b();
	cout << b << endl;
	update_model();
	cout << b << endl;
}

void RNN_f::update_model(){
	MatrixXf *M_arr[3] = {&V,&W,&U};	
	MatrixXf *del_M_arr[3] = {&del_V,&del_W,&del_U};
	VectorXf *V_arr[2] = {&b, &c};
	VectorXf *del_V_arr[2] = {&del_b, &del_c};
	update1(M_arr,del_M_arr, V_arr, del_V_arr,3, 2);
}

void RNN_f::grad_acc(){
	acc_W = acc_W + del_W;
	acc_U = acc_U + del_U;
	acc_V = acc_V + del_V;
	acc_b = acc_b + del_b;
	acc_c = acc_c + del_c;
}

void RNN_f::reset_acc(){
	acc_W = MatrixXf::Zero(d_h,d_h);
	acc_U = MatrixXf::Zero(d_h,d_x);
	acc_V = MatrixXf::Zero(d_y,d_h);
	acc_b = VectorXf::Zero(d_h);
	acc_c = VectorXf::Zero(d_y);    	
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
		cout << "d_h" << d_h << endl;
		del_H.col(t) = (W.transpose())*((MatrixXf )(VectorXf::Ones(d_h) -((VectorXf) H.col(t+2).array().square() )).asDiagonal())*del_H.col(t+1);
		del_H.col(t) = del_H.col(t) + (V.transpose())*del_O.col(t);
	}
}

// delta_V L = \sum_t delta_o^(t) h^t
void RNN_f::delta_v(){
	for (int t=0; t<d_t; t++){
		del_V = del_O.col(t)*(H.col(t).transpose());
	}
}


// delta_c L = sum_t delta_o
void RNN_f::delta_c(){
	for (int t=0; t<d_t; t++){
		del_c = del_c + del_O.col(t);
	}
}

// delta_W L = sum_t diag(1-(h^(t))^2)(delta_h^(t) L)(x^(t))^T
void RNN_f::delta_w(){
	cout << "HERE" << endl;
	for (int t=0; t<d_t; t++){
		MatrixXf HXT = del_H.col(t)*(X.col(t).transpose());
		cout << t <<  endl;
		VectorXf o = VectorXf::Ones(d_h);
		VectorXf s = H.col(t+1).array().square();
		cout << d_h << endl;
		cout << o << endl;
		cout << s << endl;
		MatrixXf N = (o-s).asDiagonal(); 
		cout << 2 << endl;
		del_U = del_U+  N*HXT;
	}
}

// delta_U L = sum_t diag(1-(h^(t))^2)(delta_h^(t) L)(h^(t-1))^T
void RNN_f::delta_u(){
	for (int t=0; t<d_t; t++){
		MatrixXf DHT = del_H.col(t)*(H.col(t).transpose());
		cout << t <<  endl;
		VectorXf o = VectorXf::Ones(d_h);
		VectorXf s = H.col(t+1).array().square();
		cout << d_h << endl;
		cout << o << endl;
		cout << s << endl;
		MatrixXf N = (o-s).asDiagonal(); 
		cout << 2 << endl;
		del_U = N*DHT;
	}
}

//  delta_b L = sum_t diag(1-(h^(t))^2)delta_h^(t) L
void RNN_f::delta_b(){
	for (int t=0; t<d_t; t++){
		del_b = ((MatrixXf )(VectorXf::Ones(d_h) -((VectorXf) H.col(t+1).array().square() )).asDiagonal())*del_H.col(t);
	}
}


