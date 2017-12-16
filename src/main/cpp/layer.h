#pragma once
#include <Eigen/Dense>
#include "functions.activation.h"
#include "functions.initialization.h"

using namespace Eigen;
using namespace neuralNetwork::functions;

namespace neuralNetwork
{
	namespace functions
	{
		namespace activation
		{
			class Activation;
		}
		namespace initialization
		{
			class Initialization;
		}
	}
	namespace layer
	{
		class Layer
		{
		protected:
			MatrixXf layer, output, bias, nebla_b;
			activation::Activation &a = activation::Identity();
		public:
			Layer(MatrixXf layer,
				MatrixXf outputs,
				MatrixXf bias) : Layer(layer, outputs, bias, activation::Identity()){};
			Layer(MatrixXf layer,
				MatrixXf outputs,
				MatrixXf bias,
				activation::Activation &a)
			{
				this->layer = layer;
				this->output = outputs;
				this->bias = bias;
				this->a = a;
			};
			virtual MatrixXf calculateOutputs() = 0;
			auto size()
			{
				return layer.size();
			}
			void setValues(MatrixXf values)
			{
				this->output = values;
			}
			void setLayer(MatrixXf values)
			{
				this->output = values;
			}
			MatrixXf values()
			{
				return layer;
			}
			MatrixXf outputs()
			{
				return output;
			}
		};
		class Connection
		{
		private:
			MatrixXf weights, nebla_w;
			Layer *input, *output;
			void initialize(initialization::Initialization *i);
		public:
			Connection(Layer * input, Layer * output, initialization::Initialization *i);
			void compute();
			void adjustWeights(float eta, int length);
			auto size() { return weights.size(); }
			Layer * getInput() { return input; }
			Layer * getOutput() { return output; }
			MatrixXf& getWeights() { return this->weights; }
		};
		class ConnectedLayer : public Layer
		{
		public:
			ConnectedLayer(int neurons)
				: Layer(MatrixXf(neurons, 1), 
					MatrixXf(neurons, 1),
					MatrixXf::Constant(neurons, 1, 0.1f),
					activation::Sigmoid()) {};
			ConnectedLayer(int neurons, activation::Activation &a)
				: Layer(MatrixXf(neurons, 1),
					MatrixXf(neurons, 1),
					MatrixXf::Constant(neurons, 1, 0.1f),
					a) {};
			ConnectedLayer(MatrixXf layer)
				: Layer(layer,
					MatrixXf(layer.rows(), layer.cols()),
					MatrixXf::Constant(layer.rows(), layer.cols(), 0.1f),
					activation::Sigmoid()) {};
			ConnectedLayer(MatrixXf layer, activation::Activation &a)
				: Layer(layer,
					MatrixXf(layer.rows(), layer.cols()),
					MatrixXf::Constant(layer.rows(), layer.cols(), 0.1f),
					a) {};
			void updateBiases(float eta, int length);
			MatrixXf calculateOutputs();
		};
	}
}