#pragma once
#include <Eigen/Dense>

using namespace Eigen;

namespace neuralNetwork
{
	namespace functions
	{
		namespace cost
		{
			class Cost
			{
			public:
				virtual float f(MatrixXf A, MatrixXf E) = 0;
				virtual MatrixXf der(MatrixXf A, MatrixXf E, MatrixXf der) = 0;
			};
			class MST : public Cost
			{
			public:
				float f(MatrixXf A, MatrixXf E);
				MatrixXf der(MatrixXf A, MatrixXf E, MatrixXf der);
			};
		}
	}
}