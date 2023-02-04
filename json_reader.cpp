#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include <Eigen/Core>
using json = nlohmann::json;
using namespace std;
using namespace Eigen;

void load_data(MatrixXf *X_s, MatrixXf *Y_s){
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

int main(){
	int n, d_t, d_x, d_y;
	data_shape(&n,&d_t,&d_x,&d_y);
	MatrixXf X_s[n];
	MatrixXf Y_s[n];
	for(int i=0;i<n;i++){
		X_s[i] = MatrixXf::Zero(d_x,d_t);
		Y_s[i] = MatrixXf::Zero(d_y,d_t);
	}
	cout << n << endl;
	cout << d_t << endl;
	cout << d_x << endl;
	cout << d_y << endl;
	load_data(X_s,Y_s);
	cout << X_s[0] << endl;
	return 0;
}
