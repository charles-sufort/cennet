#include "rnn_f.h"
#include <iostream>
#include <string>
#include <map>
#include <nlohmann/json.hpp>
#include <fstream>
#include <Eigen/Core>
#include <chrono>
using json = nlohmann::json;

using namespace Eigen;
using namespace std;


void load_data(MatrixXd *X_s, MatrixXd *Y_s){
	std::ifstream f("data_set.json");
	json data = json::parse(f);
	int n = data["X"].size();
	int d_x = data["X"][0].size();
	int d_y = data["Y"][0].size();
	int d_t = data["X"][0][0].size();
	for(int k=0; k<n; k++){
		for(int i=0;i<d_x;i++){
			for(int j=0;j<d_t; j++){
				X_s[k](i,j) = data["X"][0][i][j];
			}
		}
		for(int i=0;i<d_y;i++){
			for(int j=0;j<d_t; j++){
				Y_s[k](i,j) = data["Y"][0][i][j];
			}
		}
	}
}

void data_shape(int *n, int *d_t, int *d_x, int *d_y){
	std::ifstream f("data_set.json");
	json data = json::parse(f);
	*n = data["X"].size();
	*d_t = data["X"][0][0].size();
	*d_x = data["X"][0].size();
	*d_y = data["Y"][0].size();
}



int main() {

	int n, d_t, d_x, d_y;
	data_shape(&n,&d_t,&d_x,&d_y);
	MatrixXd X_s[n];
	MatrixXd Y_s[n];
	for(int i=0;i<n;i++){
		X_s[i] = MatrixXd::Zero(d_x,d_t);
		Y_s[i] = MatrixXd::Zero(d_y,d_t);
	}
	cout << n << endl;
	cout << d_t << endl;
	cout << d_x << endl;
	cout << d_y << endl;
	load_data(X_s,Y_s);
	struct cfg batch_cfg;
	INTMAP imap;
	FLOATMAP fmap;
	STRINGMAP smap;
	imap.insert(pair<string,int>("n_batch",40));
	batch_cfg.int_opts = imap;
	batch_cfg.double_opts = fmap;
	batch_cfg.str_opts = smap;
	RNN_f rnn_f(3,d_x,d_y,d_y,500,0.01,0.01); 
	cout << n << endl;
	std::vector<int> range;
	for (int i=0; i<n; i++){
		range.push_back(i);
	}
	MatrixXd X_train[1070];
	MatrixXd Y_train[1070];
	MatrixXd X_test[200];
	MatrixXd Y_test[200];
	for(int i=0;i<1070;i++){
		X_train[i] = X_s[range[i]];
		Y_train[i] = Y_s[range[i]];
	}
	for(int i=0;i<10;i++){
		X_test[i] = X_s[range[1070+i]];
		Y_test[i] = Y_s[range[1070+i]];
	}
	cout << "train test" << endl;

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	rnn_f.batch_gradient(X_train,Y_train,batch_cfg,1070);
	chrono::steady_clock::time_point end = chrono::steady_clock::now();
	double sec_time = chrono::duration_cast<std::chrono::seconds>(end - begin).count();
	cout << "Train time = " << sec_time << "[s]" << endl;

	cout << "test" << endl;
	int n_c = rnn_f.test(X_test,Y_test,10);
	cout << "n_c: " << n_c << endl;
	double acc = n_c/600.0;
	cout << "acc: " << acc << endl;
	return 0;
}
