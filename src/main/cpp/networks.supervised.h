#include "stdafx.h"
#include "layer.h"
#include "functions.cost.h"
#include "data.h"
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

namespace neuralNetwork
{
	namespace networks
	{
		namespace supervised
		{
			class NeuralNetwork
			{
			protected:
				vector<layer::Layer> layers;
				vector<layer::Connection> weights;
				functions::cost::Cost *cost;
				vector<float> trainingLoss, testLoss;
				float accuracy = 0, loss = 1;
				int iterations = 0;
			public:
				virtual void train(neuralNetwork::data::Data trainingData,
					data::Data testData,
					float eta,
					functions::cost::Cost cost) = 0;
				virtual void evaluate(data::Data testData) = 0;
				vector<float> getTrainingLoss() {
					return trainingLoss;
				}
				vector<float> getTestLoss() {
					return testLoss;
				}
				void setTrainingLoss(vector<float> trainingLoss) {
					this->trainingLoss = trainingLoss;
				}
				void setTestLoss(vector<float> testLoss) {
					this->testLoss = testLoss;
				}
				int getIterations() {
					return iterations;
				}
				void setIterations(int iterations) {
					this->iterations = iterations;
				}
				float getAccuracy() {
					return accuracy;
				}
				float getLoss() {
					return loss;
				}
			};
		}
	}
}