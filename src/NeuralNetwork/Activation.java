package NeuralNetwork;

import java.util.stream.*;

public abstract class Activation {
	private Activation() {}
	public abstract float[] f(float[] x);
	public abstract float[] der(float[] x);
	public static Activation identity()
	{
		return new Activation() {
			public float[] f(float[] x) {
				return x;
			}
			public float[] der(float[] x) {
				return Utils.arrayof(1, x.length);
			}
		};
	}
	public static Activation binaryStep()
	{
		return new Activation() {
			public float[] f(float[] x) {
				float[] A = new float[x.length];
				IntStream.range(0, x.length).forEach(
						a -> A[a] = x[a] < 0 ? 0 : 1);
				return A;
			}
			public float[] der(float[] x) {
				float[] A = new float[x.length];
				IntStream.range(0, x.length).forEach(
						a -> A[a] = x[a] != 0 ? 0 : null);
				return A;
			}
		};
	}
	public static Activation sigmoid()
	{
		return new Activation() {
			public float[] f(float[] x) {
				float[] A = new float[x.length];
				IntStream.range(0, x.length).forEach(
						a -> A[a] = Utils.sigmoid(x[a]));
				return A;			
			}
			public float[] der(float[] x) {
				float[] A = new float[x.length], s = f(x);
				IntStream.range(0, x.length).forEach(
						a -> A[a] = Utils.sigmoid(s[a])*(1-Utils.sigmoid(s[a])));
				return A;
			}
		};
	}
	public static Activation reLu()
	{
		return new Activation() {
			public float[] f(float[] x) {
				float[] A = new float[x.length];
				IntStream.range(0, x.length).forEach(
						a -> A[a] = x[a] < 0 ? 0 : x[a]);
				return A;
			}
			public float[] der(float[] x) {
				return binaryStep().der(x);
			}
		};
	}
	public static Activation leakyReLu()
	{
		return new Activation() {
			public float[] f(float[] x) {
				float[] A = new float[x.length];
				IntStream.range(0, x.length).forEach(
						a -> A[a] = x[a] < 0 ? 0.01f*x[a] : x[a]);
				return A;			
			}
			public float[] der(float[] x) {
				float[] A = new float[x.length];
				IntStream.range(0, x.length).forEach(
						a -> A[a] = x[a] < 0 ? 0.01f : 1);
				return A;	
			}
		};
	}
	public static Activation pReLu(float y)
	{
		return new Activation() {
			public float[] f(float[] x) {
				float[] A = new float[x.length];
				IntStream.range(0, x.length).forEach(
						a -> A[a] = x[a] < 0 ? y*x[a] : x[a]);
				return A;			
			}
			public float[] der(float[] x) {
				float[] A = new float[x.length];
				IntStream.range(0, x.length).forEach(
						a -> A[a] = x[a] < 0 ? y : 1);
				return A;	
			}
		};
	}
	public static Activation tanH()
	{
		return new Activation() {
			public float[] f(float[] x) {
				float[] A = new float[x.length];
				IntStream.range(0, x.length).forEach(
						a -> A[a] = Utils.tanH(x[a]));
				return A;			
			}
			public float[] der(float[] x) {
				float[] A = new float[x.length];
				IntStream.range(0, x.length).forEach(
						a -> A[a] = (float) (1-Math.pow(Utils.tanH(x[a]), 2)));
				return A;	
			}
		};
	}
	public static Activation softMax()
	{
		return new Activation () {
			public float[] f(float[] x) {
				float[] exp = Utils.exps(x), A = new float[x.length];
				float sum = Utils.sum(exp);
				for(int a = 0; a < x.length; a++)
					A[a] = exp[a]/sum;
				return A;
			}
			public float[] der(float[] x) {
				return null;
			}
		};
	}
	public static Activation softPlus()
	{
		return new Activation () {
			public float[] f(float[] x) {
				float[] A = new float[x.length];
				IntStream.range(0, x.length).forEach(
						a -> A[a] = (float) Math.log(1+Math.exp(x[a])));
				return A;
			}
			public float[] der(float[] x) {
				return sigmoid().f(x);
			}
		};
	}
}