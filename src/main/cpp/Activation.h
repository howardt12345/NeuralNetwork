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
				virtual MatrixXf f(MatrixXf x) = 0;
				virtual MatrixXf der(MatrixXf x) = 0;
			};
			static class Identity : public Activation
			{
			public:
				MatrixXf f(MatrixXf x);
				MatrixXf der(MatrixXf x);
			} identity;
			static class BinaryStep : public Activation
			{
			public:
				MatrixXf f(MatrixXf x);
				MatrixXf der(MatrixXf x);
			} binaryStep;
			static class Sigmoid : public Activation
			{
			public:
				MatrixXf f(MatrixXf x);
				MatrixXf der(MatrixXf x);
			} sigmoid;
			static class ReLu : public Activation
			{
			public:
				MatrixXf f(MatrixXf x);
				MatrixXf der(MatrixXf x);
			} reLu;
			static class LeakyReLu : public Activation
			{
			public:
				MatrixXf f(MatrixXf x);
				MatrixXf der(MatrixXf x);
			} leakyReLu;
			static class PReLu : public Activation
			{
			private:
				float α;
			public:
				PReLu() { α = 0.01f; }
				PReLu(float α) { this->α = α ; }
				MatrixXf f(MatrixXf x);
				MatrixXf der(MatrixXf x);
			} pReLu;
		}
	}
}