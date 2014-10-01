#ifndef _DynamicTimeWarping_
#define _DynamicTimeWarping_

#include <vector>

using namespace std;

class DTW
{
	vector< vector<double> >m_globle_cost;
	vector< vector<double> >m_dist;

	double m_top, m_mid, m_bot, m_cheapest, m_total;
	vector< vector<int> > m_move;
	vector< vector<int> > m_warp;
	vector< vector<int> > m_temp;

	unsigned int I, X, Y, n, i, j, k;
	unsigned int m_seq1_length;
	unsigned int m_seq2_length;
	unsigned int m_dims;

	unsigned int debug; /* debug flag */
	bool m_show_debug_info;

	vector< vector<double> > m_seq1, m_seq2; /*now 2 dimensional*/

	FILE *file1, *file2, *glob, *debug_file, *output_file;	
public:
	DTW();
	DTW( vector< vector<double> >& in_seq1, vector< vector<double> >& in_seq2, int dim );
	~DTW();
	void Release();

	void Initialise(vector< vector<double> >& in_seq1, vector< vector<double> >& in_seq2, int dim);
	void ComputeLoaclCostMatrix();
	double DTWDistance1Step();
	// same as the above function, but excluding the extreme situation of optimal path being vertical or horizontal line, or edges
	double DTWDistance1StepNoEdges();
};

#endif	//_DynamicTimeWarping_
