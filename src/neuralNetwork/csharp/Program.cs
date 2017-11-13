using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace NeuralNetwork
{
	class Program
	{
		public static void Main()
		{
			/*int runs = 10, size = 206;
			Matrix[] test = new Matrix[size];
			for (int a = 0; a < test.Length; a++)
			{
				Console.WriteLine("Reading matrix{0}.txt", a);
				test[a] = new Matrix(@"C:\Shared\NeuralNetwork\TextMatrix\matrix" + a + ".txt");
			}
			float[] results = new float[test.Length-1], best = new float[test.Length - 1];
			float average = 0;
			for (int a = 0; a < size-1; a++)
			{
				best[a] = float.MaxValue;
			}
			for (int i = 0; i < runs; i++)
			{
				Console.WriteLine("Run {0}", (i+1));
				Stopwatch s = new Stopwatch();
				s.Start();
				for (int a = 0; a < test.Length-1; a++)
				{
					Matrix m1 = test[a];
					Matrix m2 = test[a+1];
					Stopwatch sw = new Stopwatch();
					Console.WriteLine("Multiplying Matrix {0} and Matrix {1}", a, (a+1));
					sw.Start();
					Matrix m = m1 * m2;
					sw.Stop();
					Console.WriteLine("Done in {0}ms", sw.Elapsed.TotalMilliseconds);
					results[a] += (float)sw.Elapsed.TotalMilliseconds;
					if (sw.Elapsed.TotalMilliseconds < best[a]) best[a] = (float)sw.Elapsed.TotalMilliseconds;
				}
				s.Stop();
				Console.WriteLine("Run {0} has been running for {1} seconds", (i+1), (s.ElapsedMilliseconds/1000));
				average += (float)s.Elapsed.TotalSeconds;
			}
			Console.WriteLine("*****");
			Console.WriteLine("Results: ");
			Console.WriteLine("Runs: {0} \nSample size: {1} matrices", runs, size);
			Console.WriteLine("*****");
			for (int a = 0; a < size-1; a++)
			{
				Console.WriteLine("Average time for Matrix {0} and {1}: {2} ms", a, (a+1), (results[a]/size));
			}
			Console.WriteLine("*****");
			for (int a = 0; a < size-1; a++)
			{
				Console.WriteLine("Best time for Matrix {0} and {1}: {2} ms", a, (a + 1), best[a]);
			}
			Console.WriteLine("*****");
			Console.WriteLine("Average time for runs: {0} seconds", (average/runs));*/
			Console.WriteLine("Hello World");
			Console.ReadLine();
		}
	}
}
