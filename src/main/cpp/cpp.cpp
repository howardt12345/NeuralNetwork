// cpp.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include "neuralNetwork.h"

using namespace neuralNetwork;
using namespace std;

#include <Eigen/Dense>
using Eigen::MatrixXd;

int main()
{
	functions::activation::Sigmoid a;
	MatrixXd m(2, 2);
	m(0, 0) = 3;
	m(1, 0) = 2.5;
	m(0, 1) = -1;
	m(1, 1) = m(1, 0) + m(0, 1);
	std::cout << m << std::endl;
}