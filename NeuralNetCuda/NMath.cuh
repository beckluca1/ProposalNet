#include "header.cuh"

class Tensor
{
	public:
		int order;
		int size;

		vector<int> dimensionSizes;

		vector<float> values;

		Tensor(float in_value);

		Tensor(int in_size, vector<float> in_values);

		Tensor(int in_width, int i_height, vector<float> in_values);

		Tensor(int in_order, vector<int> in_dimensionSizes, vector<float> in_values);
};