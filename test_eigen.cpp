#include <Eigen/Core>
#include <iostream>

using namespace std;
using namespace Eigen;

void get_rand( MatrixXd *M){
	MatrixXd R = Matrix2d::Random();
	cout << M << endl;
	*M = R;
	cout << *M << endl;
}

void fill_array( int *l, int n){
	for (int i=0;i<n;i++){
		l[i] = i;
	}
}

int main(){

	MatrixXd M = MatrixXd::Random(2,2);
	VectorXd v = VectorXd::Zero(2);
	v(0) = 1;
	VectorXd h = M*v;
	MatrixXd D = v.asDiagonal();
	cout << h << endl;
	cout << M << endl;
	cout << M.diagonal() << endl;
	cout << M(0,1) << endl;
	cout << M.col(0) << endl;
	cout << v << endl;
	cout << M*v << endl;
	cout << ((MatrixXd) (VectorXd::Ones(2) - ((VectorXd) h.array().square())).asDiagonal()) - MatrixXd::Zero(2,2)<<endl;
	cout << (v.asDiagonal())*M << endl;
//	cout << v.asDiagonal() << endl;
	return 0;
}
