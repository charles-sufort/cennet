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
		MatrixXf W;
		MatrixXf U;
		MatrixXf V;
		MatrixXf A;
		MatrixXf E;
		MatrixXf H;
		MatrixXf O;
		MatrixXf Y_bar;
		MatrixXf X;
		MatrixXf Y;
		VectorXf b;
		VectorXf c;
		MatrixXf del_W;
		MatrixXf del_U;
		MatrixXf del_V;
		MatrixXf del_A;
		MatrixXf del_H;
		MatrixXf del_O;
		VectorXf del_b;
		VectorXf del_c;
		MatrixXf acc_W;
		MatrixXf acc_U;
		MatrixXf acc_V;
		MatrixXf acc_H;
		MatrixXf acc_O;
		VectorXf acc_b;
		VectorXf acc_c;
	public:
		RNN_f(int d_t,int d_x,int d_y, int d_h, int P_max_iters, float P_epoch, float P_tol);
//		void get_grad(VectorXf);
//		void update();
//		void train(MatrixXf[] X_s, VectorXf[] Y_s, int max_iters, float epoch, float tol, int method, struct cfg train_cfg);
//		void test(MatrixXf[] X_s, VectorXf[] Y_s);
		void forward_propagation(MatrixXf X);
		void backward_propagation(MatrixXf X, MatrixXf Y);
		void grad_acc();		
		void reset_acc();
		void update_model();
		void delta_o();
		void delta_h(); 
		void delta_v();
		void delta_c();
		void delta_u();
		void delta_w();
		void delta_b();
};
