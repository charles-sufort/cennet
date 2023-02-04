#include <Eigen/Core>
#include <iostream>

using namespace std;
using namespace Eigen;

VectorXf softmax(VectorXf x){
	VectorXf x_exp = x.array().exp();
	float s = x_exp.array().sum();
	x_exp = (1/s)*x_exp.array();
	return x_exp;
}

void get_rand( MatrixXf *M){
	MatrixXf R = Matrix2f::Random();
	cout << M << endl;
	*M = R;
	cout << *M << endl;
}

void fill_array( int *l, int n){
	for (int i=0;i<n;i++){
		l[i] = i;
	}
}

void add_one(

int main(){
	MatrixXf M = MatrixXf::Random(2,2);
	VectorXf v = VectorXf::Zero(2);
	v(0) = 1;
	VectorXf h = M*v;
	MatrixXf D = v.asDiagonal();
	cout << h << endl;
	cout << softmax(h) << endl;;
	cout << M << endl;
	cout << M.diagonal() << endl;
	cout << M(0,1) << endl;
	cout << M.col(0) << endl;
	cout << v << endl;
	cout << M*v << endl;
	cout << ((MatrixXf) (VectorXf::Ones(2) - ((VectorXf) h.array().square())).asDiagonal()) - MatrixXf::Zero(2,2)<<endl;
	cout << (v.asDiagonal())*M << endl;
//	cout << v.asDiagonal() << endl;
	return 0;
}
