#pragma once
#include <unordered_map>

using namespace std;

namespace neuralNetwork
{
	namespace data
	{
		class Data
		{
		protected:
			unordered_map<float*, float*> data;
		public:
			Data();
		};
		class Dataset : public Data
		{
		public:
			Dataset();
		};
	}
}