// cpp.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include "neuralNetwork.h"

using namespace neuralNetwork;
using namespace std;

#include <Eigen/Dense>
using Eigen::MatrixXf;



int main()
{
	functions::activation::Swish a(1);
	MatrixXf m(1, 15);
	m << -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7;
	cout << a.f(m) << endl;
	cout << a.der(m) << endl;
}