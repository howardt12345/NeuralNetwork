// cpp.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include "neuralNetwork.h"
#define size(x) *(&x + 1) - x

using namespace neuralNetwork;
using namespace std;


void printarray(float arg[], int length);
int main()
{
	cout << "hello world";
	float f[6] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
	printarray(f, size(f));
	cout << '\n';
	cout << size(f) << '\n';
	//Matrix m(f);
	//cout << m.getRows();
	//cout << m.getColumns();
	//cout << m.length();
	//cout << '\n';

	//m.print();
    return 0;
}
void printarray(float arg[], int length) {
	for (int n = 0; n<length; ++n)
		cout << arg[n] << ' ';
	cout << '\n';
}