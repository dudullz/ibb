#include "DynamicTimeWarping.h"

#define VERY_BIG  (1e30)
#define MIN(a,b) (a<b?a:b)  

DTW::DTW(vector< vector<double> >& in_seq1, vector< vector<double> >& in_seq2, int in_dim )
{
	m_seq1_length = in_seq1.size();
	m_seq2_length = in_seq2.size();
	m_dims = in_dim;

	m_dist.resize( m_seq1_length );
	for (int i = 0; i < m_seq1_length; ++i)
		m_dist[i].assign(m_seq2_length, 0.0);

	m_globle_cost.resize(m_seq1_length);
	for (int i = 0; i < m_seq2_length; ++i)
		m_globle_cost[i].assign(m_seq2_length, 0.0);

	m_seq1 = in_seq1;
	m_seq2 = in_seq2;
}


DTW::~DTW()
{
}


void DTW::ComputeLoaclCostMatrix()
{
	for (int i = 0; i < m_seq1_length; ++i)
		for (int j = 0; j < m_seq2_length; ++j)
		{
			double tmpDist = 0.0;
			for (int d = 0; d < m_dims; ++d)
			{
				tmpDist = tmpDist + (m_seq1[i][d] - m_seq2[j][d]) * (m_seq1[i][d] - m_seq2[j][d]);
			}
			m_dist[i][j] = tmpDist;
		}
}

double DTW::DTWDistance1Step()
{
	m_globle_cost[0][0] = m_dist[0][0];
	// the only path for 1st row is the horizontal line
	for (int j = 1; j < m_seq2_length; ++j)
		m_globle_cost[0][j] = m_globle_cost[0][j - 1] + m_dist[0][j];
	// the only path for 1st column is a vertical line
	for (int i = 1; i < m_seq1_length; ++i)
		m_globle_cost[i][0] = m_globle_cost[i-1][0] + m_dist[i][0];

	for (int i = 1; i < m_seq1_length; ++i)
		for (int j = 1; j < m_seq2_length; ++j)
		{
			m_globle_cost[i][j] = MIN( MIN(m_globle_cost[i-1][j], m_globle_cost[i][j-1]), m_globle_cost[i-1][j-1]) + m_dist[i][j];
		}

	return m_globle_cost[m_seq1_length-1][m_seq2_length-1];
}

// same as the above function, but excluding the extreme situation of optimal path being vertical or horizontal line, or edges
double DTW::DTWDistance1StepNoEdges()
{
	m_globle_cost[0][0] = m_dist[0][0];
	// the only path for 1st row is the horizontal line
	for (int j = 1; j < m_seq2_length; ++j)
		m_globle_cost[0][j] = VERY_BIG;
	// the only path for 1st column is a vertical line
	for (int i = 1; i < m_seq1_length; ++i)
		m_globle_cost[i][0] = VERY_BIG;

	for (int i = 1; i < m_seq1_length; ++i)
		for (int j = 1; j < m_seq2_length; ++j)
		{
		m_globle_cost[i][j] = MIN(MIN(m_globle_cost[i - 1][j], m_globle_cost[i][j - 1]), m_globle_cost[i - 1][j - 1]) + m_dist[i][j];
		}

	return m_globle_cost[m_seq1_length - 1][m_seq2_length - 1];
}