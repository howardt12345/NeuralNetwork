#pragma once
#include "neuralNetwork.h"

namespace neuralNetwork
{
	namespace functions
	{
		namespace initialization
		{
			class Initialization
			{
			public:
				virtual void initialize(layer::Connection *c) = 0;
			};
			class Zeros : public Initialization
			{
			public:
				void initialize(layer::Connection * c);
			};
			class Ones : public Initialization
			{
			public:
				void initialize(layer::Connection * c);
			};
			class Constant : public Initialization
			{
			private:
				float x;
			public:
				Constant(float x) { this->x = x; };
				void initialize(layer::Connection * c);
			};
			class RandomNormal : public Initialization
			{
			public:
				void initialize(layer::Connection * c);
			};
			class RandomUniform : public Initialization
			{
			public:
				void initialize(layer::Connection * c);
			};
		}
	}
}
