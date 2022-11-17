#include "NMath.cuh"

Tensor::Tensor(float in_value) : Tensor(0, { }, { in_value })
{
}

Tensor::Tensor(int in_size, vector<float> in_values) : Tensor(1, { in_size }, in_values)
{
}

Tensor::Tensor(int in_width, int in_height, vector<float> in_values) : Tensor(2, { in_width, in_height }, in_values)
{
}

Tensor::Tensor(int in_order, vector<int> in_dimensionSizes, vector<float> in_values)
{
	order = in_order;

	dimensionSizes = vector<int>(order);

	size = 1;

	for (int i = 0; i < in_order; i++)
	{
		dimensionSizes[i] = in_dimensionSizes[i];
		size *= in_dimensionSizes[i];
	}

	values = vector<float>(size);

	for (int i = 0; i < size; i++)
	{
		values[i] = in_values[i];
	}
}