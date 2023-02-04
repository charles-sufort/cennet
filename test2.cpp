#include <iostream>
#include <Eigen/Core>

using namespace Eigen;
using namespace std;

int main(){
	MatrixXf M;
	M = MatrixXf::Zero(2,2);
	cout << M <<endl;
	M << 3.0,2.0,1.0,1.3;
	cout << M << endl;
	return 0;
}
