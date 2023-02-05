#include <Eigen/Core>

#include "rnn.h"
using namespace std;
using namespace Eigen;

class RNN_f: public RNN{
	private:
		int d_h;
		int d_x;
		int d_y;
		int d_t;
		MatrixXd W;
		MatrixXd U;
		MatrixXd V;
		MatrixXd A;
		MatrixXd E;
		MatrixXd H;
		MatrixXd O;
		MatrixXd Y_bar;
		MatrixXd X;
		MatrixXd Y;
		VectorXd b;
		VectorXd c;
		MatrixXd del_W;
		MatrixXd del_U;
		MatrixXd del_V;
		MatrixXd del_A;
		MatrixXd del_H;
		MatrixXd del_O;
		VectorXd del_b;
		VectorXd del_c;
		MatrixXd acc_W;
		MatrixXd acc_U;
		MatrixXd acc_V;
		MatrixXd acc_H;
		MatrixXd acc_O;
		VectorXd acc_b;
		VectorXd acc_c;
	public:
		RNN_f(int P_d_t,int P_d_x,int P_d_y, int P_d_h, int P_max_iters, double P_epoch, double P_tol);
//		void get_grad(VectorXd);
//		void update();
//		void train(MatrixXd[] X_s, VectorXd[] Y_s, int max_iters, double epoch, double tol, int method, struct cfg train_cfg);
//		void test(MatrixXd[] X_s, VectorXd[] Y_s);
		void forward_propagation(MatrixXd X);
		void backward_propagation(MatrixXd X, MatrixXd Y);
		void grad_acc();		
		void reset_acc();
		void update_model(int model);
		void get_Y_pred(int **y_p_p);
		void delta_o();
		void delta_h(); 
		void delta_v();
		void delta_c();
		void delta_u();
		void delta_w();
		void delta_b();
};
