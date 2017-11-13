package Java.neuralNetwork;

import java.io.*;
import java.util.*;
import java.util.function.*;
import java.util.stream.*;

public class Utils {
	
	public static void print(float[] out)
	{
		for(int a = 0; a < out.length; a++)
			System.out.println(a+": "+out[a]);
	}
	public static void print(float[] out, String header)
	{
		System.out.println(header);
		for(int a = 0; a < out.length; a++)
			System.out.println(a+": "+out[a]);
	}
	public static void print(float[][] out) 
	{
		for (int a = 0; a < out.length; a++)
		{
			for (int b = 0; b < out[a].length; b++)
				System.out.print(out[a][b] + " ");
			System.out.println();
		}
	}
	public static float[] copy(float[] x)
	{
		float[] a = new float[x.length];
		System.arraycopy(x, 0, a, 0, x.length);
		return a;
	}
	public static float[] apply(float[] x, Function<Float, Float> f)
	{
		float[] A = copy(x);
		for(int a = 0; a < A.length; a++)
			A[a] = f.apply(x[a]);
		return A;
	}
	public static float[] arrayof(float value, int size)
	{
		float[] x = new float[size];
		IntStream.range(0, size).forEach(a -> x[a] = value);
		return x;
	}
	public static float[] add(float[] x, float[] y)
	{
		assert(x.length == y.length) : x.length + " != " + y.length;
		float[] A = new float[x.length];
		for(int a = 0; a < A.length; a++)
			A[a] = x[a] + y[a];
		return A;
	}
	public static float[] subtract(float[] x, float[] y)
	{
		assert(x.length == y.length) : x.length + " != " + y.length;
		float[] A = new float[x.length];
		for(int a = 0; a < A.length; a++)
			A[a] = x[a] - y[a];
		return A;
	}
	public static float[] subtract(float[] x, float y)
	{
		float[] A = new float[x.length];
		for(int a = 0; a < A.length; a++)
			A[a] = x[a] - y;
		return A;
	}
	public static float[] multiply(float[] x, float[] y)
	{
		assert(x.length == y.length) : x.length + " != " + y.length;
		float[] A = new float[x.length];
		for(int a = 0; a < A.length; a++)
			A[a] = x[a] * y[a];
		return A;
	}
	public static float[] divide(float[] x, float[] y)
	{
		assert(x.length == y.length) : x.length + " != " + y.length;
		float[] A = new float[x.length];
		for(int a = 0; a < A.length; a++)
			A[a] = x[a] / y[a];
		return A;
	}
	public static float[] multiply(float[] x, float y)
	{
		float[] A = new float[x.length];
		for(int a = 0; a < A.length; a++)
			A[a] = x[a] * y;
		return A;
	}
	public static float[] divide(float[] x, float y)
	{
		float[] A = new float[x.length];
		for(int a = 0; a < A.length; a++)
			A[a] = x[a] / y;
		return A;
	}
	public static void randomize(float[] x, int min, int max)
	{
		IntStream.range(0, x.length).forEach(a -> x[a] = random(min, max));
	}
	public static float random(float x, float y)
	{
		return x + (float)Math.random() * (y-x);
	}
	public static float sum(float[] x)
	{
		float tmp = 0;
		for(int a = 0; a < x.length; a++)
			tmp += x[a];
		return tmp;
	}
	public static float sum(List<Float> x)
	{
		float tmp = 0;
		for(int a = 0; a < x.size(); a++)
			tmp += x.get(a);
		return tmp;
	}
	public static float[] toFloat(int[] x)
	{
		float[] tmp = new float[x.length];
		IntStream.range(0, x.length).forEach(a -> tmp[a] = (float)x[a]);
		return tmp;
	}
	public static float[] toFloat(double[] x)
	{
		float[] tmp = new float[x.length];
		IntStream.range(0, x.length).forEach(a -> tmp[a] = (float)x[a]);
		return tmp;
	}
	public static float[][] toFloat(int[][] x)
	{
		float[][] tmp = new float[x.length][];
		for(int a = 0; a < tmp.length; a++)
			tmp[a] = toFloat(x[a]);
		return tmp;
	}
	public static float[][] toFloat(double[][] x)
	{
		float[][] tmp = new float[x.length][];
		for(int a = 0; a < tmp.length; a++)
			tmp[a] = toFloat(x[a]);
		return tmp;
	}
	public static float[] toArray(String x)
	{
		String[] in = x.split(" ");
		float[] tmp = new float[in.length];
		IntStream.range(0, tmp.length).forEach(a -> tmp[a] = Float.parseFloat(in[a]));
		return tmp;
	}
	public static String asString(int[][] x)
	{
		StringBuffer sb = new StringBuffer();
		for(int a = 0; a < x.length; a++)
			for(int b = 0; b < x[a].length; b++)
				sb.append(x[a][b] + " ");
		return sb.toString();
	}
	public static float sigmoid(float x)
	{
		return 1/(1+exp(-x));
	}
	public static float tanH(float x)
	{
		return (float) Math.tanh(x);
	}
	public static float exp(float x)
	{
		return (float)Math.exp(x);
	}
	public static float ln(float x)
	{
		return (float)Math.log(x);
	}
	public static float[] exps(float[] x)
	{
		float[] A = new float[x.length];
		IntStream.range(0, x.length).forEach(a -> A[a] = exp(x[a]));
		return A;
	}
	public static void setOutput(String filename)
	{
		try {
			PrintStream out = new PrintStream(new FileOutputStream(filename));
			System.setOut(out);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
}
