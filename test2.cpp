#include <iostream>
#include <Eigen/Core>

using namespace Eigen;
using namespace std;

int main(){
	std::vector<int> range;
	int n = 100;
	for (int i=0; i<n; i++) range.push_back(i);
	cout << range[0] << endl;
	random_shuffle(range.begin(), range.end());
	cout << range[0] << endl;
	return 0;
}
