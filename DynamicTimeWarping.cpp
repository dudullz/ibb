#include "DynamicTimeWarping.h"


DTW::DTW(vector< vector<float> >& in_seq1, vector< vector<float> >& in_seq2, int in_dim )
{
	m_seq1_length = in_seq1.size();
	m_seq2_length = in_seq2.size();
	m_dims = in_dim;
}


DTW::~DTW()
{
}


void DTW::ComputeLoaclCostMatrix()
{
	for (int i = 0; i < m_seq1_length; ++i)
		for (int j = 0; j < m_seq2_length; ++j)
		{
			
		}
}

void DTW::DTWDistance()
{

}