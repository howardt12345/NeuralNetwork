#include "stdafx.h"
#include "neuralNetwork.h"
#include "Activation.h"

using namespace neuralNetwork::functions::activation;

/*
MatrixXf [name]::f(MatrixXf x)
{
return MatrixXf(1, 1);
}
MatrixXf [name]::der(MatrixXf x)
{
return MatrixXf(1, 1);
}
*/


MatrixXf Identity::f(MatrixXf x)
{
	return MatrixXf();
}
MatrixXf Identity::der(MatrixXf x)
{
	return MatrixXf();
}
MatrixXf BinaryStep::f(MatrixXf x)
{
	return MatrixXf(1, 1);
}
MatrixXf BinaryStep::der(MatrixXf x)
{
	return MatrixXf(1, 1);
}
MatrixXf Sigmoid::f(MatrixXf x)
{
	return MatrixXf(1, 1);
}
MatrixXf Sigmoid::der(MatrixXf x)
{
	return MatrixXf(1, 1);
}
MatrixXf ReLu::f(MatrixXf x)
{
	return MatrixXf(1, 1);
}
MatrixXf ReLu::der(MatrixXf x)
{
	return MatrixXf(1, 1);
}
MatrixXf LeakyReLu::f(MatrixXf x)
{
	return MatrixXf(1, 1);
}
MatrixXf LeakyReLu::der(MatrixXf x)
{
	return MatrixXf(1, 1);
}
MatrixXf PReLu::f(MatrixXf x)
{
	return MatrixXf(1, 1);
}
MatrixXf PReLu::der(MatrixXf x)
{
	return MatrixXf(1, 1);
}
