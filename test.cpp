#include <map>
#include <iostream>
#include <string>
#include <Eigen/Core>

using namespace Eigen;
using namespace std;

typedef multimap<string, int> INTMAP;
typedef multimap<string, float> FLOATMAP;
typedef multimap<string, string> STRINGMAP; 

struct cfg {
	INTMAP int_opts;
	FLOATMAP float_opts;
	STRINGMAP str_opts;
};

void cfg_print(){

}

int main(){
	INTMAP i_map;
	FLOATMAP f_map;
	STRINGMAP s_map;
	string A = "A";
	string B = "B";
	string C = "C";
	i_map.insert(pair<string,int>(A,3));
	f_map.insert(pair<string,float>(B,3.0));
	s_map.insert(pair<string,string>(C,"third"));
	struct cfg cfg1;
	cfg1.int_opts = i_map;
	cfg1.float_opts = f_map;
	cfg1.str_opts = s_map;
	cout << cfg1.int_opts.find(A)->second << endl;
	int i1 = cfg1.int_opts.find(A)->second;
	return 0;
}

