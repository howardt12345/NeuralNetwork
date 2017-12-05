#pragma once
#include <Eigen/Dense>
#include "data.h"
#include <vector>

using namespace std;
using namespace neuralNetwork::data;

namespace neuralNetwork
{
	namespace training
	{
		class Training
		{
		protected:
			vector<float> trainingLoss, testLoss;
			Dataset *trainingData = NULL, *testData = NULL;
		public:
			Training(Dataset * training, Dataset * test)
			{
				this->trainingData = training;
				this->testData = test;
			}
			virtual void train(int epochs) = 0;
		};
	}
}