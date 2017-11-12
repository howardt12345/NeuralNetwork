using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Diagnostics;
using System.Globalization;
using System.Threading.Tasks;
using System.Collections.Generic;

namespace NeuralNetwork
{
	public class Matrix
	{
		private float[] matrix = new float[16];
		private int ROWS = 4, COLUMNS = 4;
		public Matrix(int row, int column)
		{
			matrix = new float[row * column];
			for (int a = 0; a < row; a++)
			{
				for (int b = 0; b < column; b++)
					this[a, b] = a == b ? 1 : 0;
			}
			ROWS = row;
			COLUMNS = column;
		}
		public Matrix(String file)
		{

			.string[][] result = File
			.ReadLines(file)
			.Select(line => line.Split(' '))
			.ToArray();
			matrix = new float[result.Length*result[0].Length];
			int i = 0;
			ROWS = result.Length;
			COLUMNS = result[0].Where(x => !string.IsNullOrEmpty(x)).ToArray().Length;
			for (int a = 0; a < result.Length; a++)
			{
				string[] inner = result[a].Where(x => !string.IsNullOrEmpty(x)).ToArray();
				for (int b = 0; b < inner.Length; b++)
				{
					float.TryParse(inner[b], out matrix[i++]);
				}
			}
		}
		public static Matrix identity(int rows, int columns)
		{
			Matrix m = new Matrix(rows, columns);
			for (int a = 0; a < m.rows; a++)
			{
				for (int b = 0; b < m.columns; b++)
					m[a, b] = a == b ? 1 : 0;
			}
			return m;
		}
		private static Matrix add(Matrix m1, Matrix m2)
		{
			if (m1.rows == m2.rows && m1.columns == m2.columns)
			{
				Matrix m = new Matrix(m1.rows, m1.columns);
				for (int a = 0; a < m1.rows; a++)
					for (int b = 0; b < m1.columns; b++)
						m[a, b] = m1[a, b] + m2[a, b];
				return m;
			}
			else return null;
		}
		private static Matrix subtract(Matrix m1, Matrix m2)
		{
			if (m1.rows == m2.rows && m1.columns == m2.columns)
			{
				Matrix m = new Matrix(m1.rows, m1.columns);
				for (int a = 0; a < m1.rows; a++)
					for (int b = 0; b < m1.columns; b++)
						m[a, b] = m1[a, b] - m2[a, b];
				return m;
			}
			else return null;
		}
		private static Matrix multiply(Matrix m1, Matrix m2)
		{
			if (m1.columns != m2.rows)
				return null;
			else
			{
				Matrix m = new Matrix(m1.rows, m2.columns);
				for (int b = 0; b < m2.columns; b++)
				{
					for (int a = 0; a < m1.rows; a++)
					{
						float sum = 0;
						for (int c = 0; c < m2.rows; c++)
						{
							float elementA = m1.matrix[a * m1.columns + c];
							float elementB = m2.matrix[c * m2.columns + b];
							sum += elementA * elementB;
						}
						m[a, b] = sum;
					}
				}
				return m;
			}
		}
		public static Matrix operator +(Matrix m1, Matrix m2) => add(m1, m2);
		public static Matrix operator -(Matrix m1, Matrix m2) => subtract(m1, m2);
		public static Matrix operator *(Matrix m1, Matrix m2) => multiply(m1, m2);
		public void print()
		{
			for (int a = 0; a < rows; a++)
			{
				for (int b = 0; b < columns; b++)
					Console.Write("{0}{1}", this[a, b], b == (COLUMNS - 1) ? "" : " ");
				Console.WriteLine();
			}
		}
		public float this[int row, int column]
		{
			get
			{
				return matrix[row*COLUMNS + column];
			}
			set
			{
				matrix[row*COLUMNS + column] = value;
			}
		}
		public float this[int index]
		{
			get
			{
				return matrix[index];
			}
			set
			{
				matrix[index] = value;
			}
		}
		public int rows => ROWS;
		public int columns => COLUMNS;
	}
}
