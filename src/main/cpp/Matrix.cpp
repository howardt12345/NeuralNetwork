#include "stdafx.h"
#include "neuralNetwork.h"
#define size(x) *(&x + 1) - x

using namespace neuralNetwork;

Matrix::Matrix(float matrix[])
{
	std::cout << size(matrix);
	this->matrix = new float[size(matrix)];
	for (int i = 0; i < size(matrix); i++)
	{
		this->matrix[i] = matrix[i];
	}
	this->rows = size(matrix);
	this->columns = 1;
}
Matrix::Matrix(float *matrix, int rows, int columns)
{
	this->matrix = new float[rows*columns];
	this->rows = rows;
	this->columns = columns;
	int k = 0;
	for (int i = 0; i < rows*columns; i++)
	{
		this->matrix[i] = matrix[i];
	}
}
Matrix::Matrix(int rows, int columns)
{
	this->rows = rows;
	this->columns = columns;
	this->matrix = new float[rows*columns];
}
Matrix::Matrix(int length)
{
	this->rows = length;
	this->columns = 1;
	this->matrix = new float[length];
}
Matrix Matrix::identity(int rows, int columns)
{
	Matrix m(rows, columns);
	for (int a = 0; a < m.rows; a++)
	{
		for (int b = 0; b < m.columns; b++)
			m.set(a == b ? 1.0f : 0.0f, a, b);
	}
	return m;
}
Matrix Matrix::diagonal(float values[])
{
	Matrix m(size(values));
	return m;
}
void Matrix::print()
{
	for (int a = 0; a < rows; a++)
	{
		for (int b = 0; b < columns; b++)
			std::cout << get(a, b) << " ";
		std::cout << '\n';
	}
}