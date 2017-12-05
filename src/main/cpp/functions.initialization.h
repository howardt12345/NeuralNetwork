#pragma once

namespace neuralNetwork
{
	namespace functions
	{
		namespace initialization
		{
			class Initialization
			{
				virtual float f(float x, float y) = 0;
				virtual void initialize() = 0;
			};
		}
	}
}