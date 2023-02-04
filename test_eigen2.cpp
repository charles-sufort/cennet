#include <Eigen/Core>
#include <iostream>

using namespace std;
using namespace Eigen;

void add_one(MatrixXf **Ms,int n){
	MatrixXf O = MatrixXf::Ones(2,2);
	for (int i=0; i<n; i++){
		(*Ms[i]) = (*Ms[i]) + O;
	}
}

int main(){
	MatrixXf M1 = MatrixXf::Random(2,2);
	MatrixXf M2 = MatrixXf::Random(2,2);
	MatrixXf M3 = MatrixXf::Random(2,2);
	MatrixXf *Ms[3] = {&M1 , &M2, &M3};
	cout << M2 << endl;
	add_one(Ms, 3);
	cout << M2 << endl;
	return 0;
}
