#include "stdafx.h"
#include "neuralNetwork.h"
#include "functions.activation.h"

using namespace neuralNetwork::functions::activation;
using namespace std;

MatrixXf Identity::f(MatrixXf x)
{
	return x;
}
MatrixXf Identity::der(MatrixXf x)
{
	return MatrixXf::Ones(x.rows(), x.cols());
}

float Binary::activation(float x)
{
	return x > 0 ? 1.0f : 0.0f;
}
float Binary::derivative(float x)
{
	return x != 0 ? 0.0f : NAN;
}
MatrixXf Binary::f(MatrixXf x)
{
	return x.unaryExpr(ptr_fun(&Binary::activation));
}
MatrixXf Binary::der(MatrixXf x)
{
	return x.unaryExpr(ptr_fun(&Binary::derivative));
}

float Sigmoid::activation(float x)
{
	return 1 / (1 + exp(-x));
}
float Sigmoid::derivative(float x)
{
	return activation(x)*(1 - activation(x));
}
MatrixXf Sigmoid::f(MatrixXf x)
{
	return x.unaryExpr(ptr_fun(&Sigmoid::activation));
}
MatrixXf Sigmoid::der(MatrixXf x)
{
	return x.unaryExpr(ptr_fun(&Sigmoid::derivative));
}

float ReLu::activation(float x)
{
	return x > 0 ? x : 0.0f;
}
MatrixXf ReLu::f(MatrixXf x)
{
	return x.unaryExpr(ptr_fun(&ReLu::activation));
}
MatrixXf ReLu::der(MatrixXf x)
{
	return Binary().f(x);
}

float LeakyReLu::activation(float x)
{
	return x > 0 ? x : 0.01f*x;
}
float LeakyReLu::derivative(float x)
{
	return x > 0 ? 1 : 0.01f;
}
MatrixXf LeakyReLu::f(MatrixXf x)
{
	return x.unaryExpr(ptr_fun(&LeakyReLu::activation));
}
MatrixXf LeakyReLu::der(MatrixXf x)
{
	return x.unaryExpr(ptr_fun(&LeakyReLu::derivative));
}

struct PReLuActivation
{
	float operator()(float x, float a) const
	{
		return x > 0 ? x : a*x;
	}
};
struct PReLuDerivative
{
	float operator()(float x, float a) const
	{
		return x > 0 ? 1 : a;
	}
};
MatrixXf PReLu::f(MatrixXf x)
{
	return x.binaryExpr(MatrixXf::Constant(x.rows(), x.cols(), α), 
		PReLuActivation());
}
MatrixXf PReLu::der(MatrixXf x)
{
	return x.binaryExpr(MatrixXf::Constant(x.rows(), x.cols(), α), 
		PReLuDerivative());
}

float TanH::derivative(float x)
{
	return 1 - pow(tanhf(x), 2);
}
MatrixXf TanH::f(MatrixXf x)
{
	return x.array().tanh();
}
MatrixXf TanH::der(MatrixXf x)
{
	return x.unaryExpr(ptr_fun(&TanH::derivative));
}

struct SoftMaxActivation
{
	float operator()(float x, float a) const
	{
		return x / a;
	}
};

MatrixXf SoftMax::f(MatrixXf x)
{
	MatrixXf exp = x.array().exp();
	MatrixXf sums = MatrixXf::Constant(x.rows(), x.cols(), exp.sum());
	return exp.binaryExpr(sums, SoftMaxActivation());
}
MatrixXf SoftMax::der(MatrixXf x)
{
	MatrixXf f = SoftMax().f(x),
		δ = MatrixXf::Identity(x.rows()*x.cols(), x.rows()*x.cols()),
		A = MatrixXf(x.rows(), x.cols());
	for (int a = 0; a < x.rows(); a++)
		for (int b = 0; b < x.cols(); b++)
			A(a) = f(a)*δ(a, b) - f(b);
	return A;
}

MatrixXf SoftPlus::f(MatrixXf x)
{
	return (x.array().exp() + 1).array().log();
}
MatrixXf SoftPlus::der(MatrixXf x)
{
	return Sigmoid().f(x);
}

float Swish::derivative(float x)
{
	return x > 0 ? x : 0.01f;
}
MatrixXf Swish::f(MatrixXf x)
{
	return x.array() * Sigmoid().f((x.array() * β)).array();
}
MatrixXf Swish::der(MatrixXf x)
{
	return (β * f(x).array()) + Sigmoid().f((β * x.array())).array()*(1 - β * f(x).array());
}