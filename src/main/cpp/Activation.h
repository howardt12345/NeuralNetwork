#pragma once
#include "neuralNetwork.h"
#include <Eigen/Dense>

using namespace Eigen;

namespace neuralNetwork
{
	namespace functions
	{
		namespace activation
		{
			class Activation
			{
			public:
				virtual MatrixXf f(MatrixXf x);
				virtual MatrixXf der(MatrixXf x);
			};
			class Identity : public Activation
			{
			public:
				MatrixXf f(MatrixXf x);
				MatrixXf der(MatrixXf x);
			};
			class Binary : public Activation
			{
			private:
				static float activation(float x);
				static float derivative(float x);
			public:
				MatrixXf f(MatrixXf x);
				MatrixXf der(MatrixXf x);
			};
			class Sigmoid : public Activation
			{
			private:
				static float activation(float x);
				static float derivative(float x);
			public:
				MatrixXf f(MatrixXf x);
				MatrixXf der(MatrixXf x);
			};
			class ReLu : public Activation
			{
			private:
				static float activation(float x);
			public:
				MatrixXf f(MatrixXf x);
				MatrixXf der(MatrixXf x);
			};
			class LeakyReLu : public Activation
			{
			private:
				static float activation(float x);
				static float derivative(float x);
			public:
				MatrixXf f(MatrixXf x);
				MatrixXf der(MatrixXf x);
			};
			class PReLu : public Activation
			{
			private:
				float α;
			public:
				PReLu(float α) { this->α = α; }
				MatrixXf f(MatrixXf x);
				MatrixXf der(MatrixXf x);
			};
			class TanH : public Activation
			{
			private:
				static float derivative(float x);
			public:
				MatrixXf f(MatrixXf x);
				MatrixXf der(MatrixXf x);
			};
			class SoftMax : public Activation
			{
			public:
				MatrixXf f(MatrixXf x);
				MatrixXf der(MatrixXf x);
			};
			class SoftPlus : public Activation
			{
			public:
				MatrixXf f(MatrixXf x);
				MatrixXf der(MatrixXf x);
			};
			
			class Swish : public Activation
			{
			private:
				float derivative(float x);
				float β;
			public:
				Swish(float β) { this->β = β; }
				MatrixXf f(MatrixXf x);
				MatrixXf der(MatrixXf x);
			};
		}
	}
}