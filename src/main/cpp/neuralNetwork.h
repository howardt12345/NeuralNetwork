#pragma once
#include "stdafx.h"
#include <iostream>

#define size(x) (sizeof(x)/sizeof((x)[0]))

namespace neuralNetwork
{
	namespace functions 
	{
		class Activation
		{
		};
	}
	namespace layer 
	{
	}
	namespace networks
	{
		namespace supervised 
		{
		}
	}
	namespace training 
	{
	}
	class Utils 
	{
	};
	class Matrix 
	{
	protected:
		float* matrix;
		int rows, columns;
	public:
		Matrix(float matrix[]);
		Matrix(float *matrix, int rows, int columns);
		Matrix(int rows, int columns);
		Matrix(int length);
		static Matrix identity(int rows, int columns);
		static Matrix diagonal(float values[]);
		float get(int row, int column) { return matrix[row*columns + column]; }
		float get(int index) { return matrix[index]; }
		void set(float value, int row, int column) { matrix[row*columns + column] = value; }
		int getRows() { return rows; }
		int getColumns() { return columns; }
		int length() { return rows*columns; }
		float* getMatrix() { return matrix; }
		void print();
		~Matrix() { delete[] matrix; }
	};
}
