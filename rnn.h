#include <Eigen/Core>
#include <map>
#include <string>


using namespace std;
using namespace Eigen;

typedef multimap<string, int> INTMAP;
typedef multimap<string, double> FLOATMAP;
typedef multimap<string, string> STRINGMAP; 

struct cfg {
	INTMAP int_opts;
	FLOATMAP double_opts;
	STRINGMAP str_opts;
};


class RNN{
	public:
		int d_t;
		int d_y;
		int max_iters;
		double epoch;
		double tol;
		RNN(int P_max_iters, double P_epoch, double P_tol, int P_d_t, int P_d_y);
		void train(MatrixXd X_S[], MatrixXd Y_S[],int method, struct cfg train_cfg, int n);
		double test(MatrixXd X_S[], MatrixXd Y_S[], int n);
		void batch_gradient(MatrixXd X_S[], MatrixXd Y_S[],  struct cfg batch_cfg, int n);
		VectorXd softmax(VectorXd x);
		void update_gd(MatrixXd **M_arr,MatrixXd **del_M_arr, VectorXd **V_arr, VectorXd **del_V_arr,int m_n, int v_n);
		virtual void grad_acc();
		virtual void reset_acc();
		virtual void get_Y_pred(int **y_p_p);
		virtual void update_model();
		virtual void forward_propagation(MatrixXd X);
		virtual void backward_propagation(MatrixXd X, MatrixXd Y);
};
