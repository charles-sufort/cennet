#include <Eigen/Core>
#include <map>
#include <string>


using namespace std;
using namespace Eigen;

typedef multimap<string, int> INTMAP;
typedef multimap<string, float> FLOATMAP;
typedef multimap<string, string> STRINGMAP; 

struct cfg {
	INTMAP int_opts;
	FLOATMAP float_opts;
	STRINGMAP str_opts;
};


class RNN{
	public:
		int max_iters;
		float epoch;
		float tol;
		RNN(int P_max_iters, float P_epoch, float P_tol);
		void train(MatrixXf X_S[], MatrixXf Y_S[],int method, struct cfg train_cfg);
		void batch_gradient(MatrixXf X_S[], MatrixXf Y_S[], int method, struct cfg batch_cfg);
		VectorXf softmax(VectorXf x);
		void update1(MatrixXf **M_arr,MatrixXf **del_M_arr, VectorXf **V_arr, VectorXf **del_V_arr,int m_n, int v_n);
};
