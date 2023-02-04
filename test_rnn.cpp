#include "rnn_f.h"
#include <iostream>
#include <string>
#include <map>
#include <Eigen/Core>

using namespace Eigen;
using namespace std;
typedef multimap<string, int> INTMAP;
typedef multimap<string, float> FLOATMAP;
typedef multimap<string, string> STRINGMAP; 

int main() {
	RNN_f rnn_f(2,2,2,2,2,0.01,0.01); 
	MatrixXf X1 = MatrixXf::Zero(2,2);
	MatrixXf X2 = MatrixXf::Zero(2,2);
	X1(0,0) = 1;
	X1(1,1) = 1;
	X2(0,1) = 1;
	X2(1,0) = 1;
	MatrixXf Y1 = MatrixXf::Zero(2,2);
	MatrixXf Y2 = MatrixXf::Zero(2,2);
	Y1(1,0) = 1;
	Y1(0,1) = 1;
	Y2(1,0) = 1;
	Y2(0,1) = 1;
	MatrixXf X[2] = {X1,X2};
	MatrixXf Y[2] = {Y1,Y2};
	cout << "here" << endl;
	rnn_f.backward_propagation(X1,Y1);
	return 0;
}
