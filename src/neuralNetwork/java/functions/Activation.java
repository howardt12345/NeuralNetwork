package neuralNetwork.java.functions;

import java.util.stream.*;

import neuralNetwork.java.*;

public abstract class Activation {
	private Activation() {}
	public abstract float[] f(float[] x);
	public abstract float[] der(float[] x);
	/** <p>The Identity activation function, given by f(x) = x.</p>
	 * <p>The derivative of this function is f'(x) = 1.</p>
	 */
	public static Activation identity()
	{
		return new Activation() {
			@Override
			public float[] f(float[] x) {
				return x;
			}
			@Override
			public float[] der(float[] x) {
				return Utils.arrayof(1, x.length);
			}
		};
	}
	/** <p>The Binary step activation function, given by f(x) = { 0 for x < 0, 1 for x >= 0.</p>
	 * <p>The derivative of this function is f'(x) = { 0 for x != 0, ? for x = 0.</p>
	 */
	public static Activation binaryStep()
	{
		return new Activation() {
			@Override
			public float[] f(float[] x) {
				float[] A = new float[x.length];
				IntStream.range(0, x.length).forEach(
						a -> A[a] = x[a] < 0 ? 0 : 1);
				return A;
			}
			@Override
			public float[] der(float[] x) {
				float[] A = new float[x.length];
				IntStream.range(0, x.length).forEach(
						a -> A[a] = x[a] != 0 ? 0 : null);
				return A;
			}
		};
	}
	/** <p>The Logistic activation function, given by f(x) = 1/(1+exp(-x)).</p>
	 * <p>The derivative of this function is f'(x) = f(x)*(1-f(x)).</p>
	 */
	public static Activation sigmoid()
	{
		return new Activation() {
			@Override
			public float[] f(float[] x) {
				float[] A = new float[x.length];
				IntStream.range(0, x.length).forEach(
						a -> A[a] = Utils.sigmoid(x[a]));
				return A;			
			}
			@Override
			public float[] der(float[] x) {
				float[] A = new float[x.length], s = f(x);
				IntStream.range(0, x.length).forEach(
						a -> A[a] = Utils.sigmoid(s[a])*(1-Utils.sigmoid(s[a])));
				return A;
			}
		};
	}
	/** <p>The Rectified linear unit (reLU) activation function, given by f(x) = max(0, x).</p>
	 * <p>The derivative of this function is f'(x) = { 0 for x < 0, 1 for x >= 0.</p>
	 */
	public static Activation reLU()
	{
		return new Activation() {
			@Override
			public float[] f(float[] x) {
				float[] A = new float[x.length];
				IntStream.range(0, x.length).forEach(
						a -> A[a] =  Math.max(0, x[a]));
				return A;
			}
			@Override
			public float[] der(float[] x) {
				return binaryStep().der(x);
			}
		};
	}
	/** <p>The Leaky ReLU activation function, given by f(x) = { 0.01x for x < 0, x for x >= 0.</p>
	 * <p>The derivative of this function is f'(x) = { 0.01 for x < 0, 1 for x >= 0.</p>
	 */
	public static Activation leakyReLu()
	{
		return new Activation() {
			@Override
			public float[] f(float[] x) {
				float[] A = new float[x.length];
				IntStream.range(0, x.length).forEach(
						a -> A[a] = x[a] < 0 ? 0.01f*x[a] : x[a]);
				return A;			
			}
			@Override
			public float[] der(float[] x) {
				float[] A = new float[x.length];
				IntStream.range(0, x.length).forEach(
						a -> A[a] = x[a] < 0 ? 0.01f : 1);
				return A;	
			}
		};
	}
	/** <p>The Parametric ReLU activation function, given by f(x) = { ax for x < 0, x for x >= 0.</p>
	 * <p>The derivative of this function is f'(x) = { a for x < 0, 1 for x >= 0.</p>
	 * @param a the coefficient of leakage.
	 */
	public static Activation pReLu(float a)
	{
		return new Activation() {
			@Override
			public float[] f(float[] x) {
				float[] A = new float[x.length];
				IntStream.range(0, x.length).forEach(
						i -> A[i] = x[i] < 0 ? a*x[i] : x[i]);
				return A;			
			}
			@Override
			public float[] der(float[] x) {
				float[] A = new float[x.length];
				IntStream.range(0, x.length).forEach(
						i -> A[i] = x[i] < 0 ? a : 1);
				return A;	
			}
		};
	}
	/** <p>The Hyperbolic tangent activation function, given by f(x) = tanh(x) = 2/(1+exp(-2x))-1.</p>
	 * <p>The derivative of this function is f'(x) = 1-f(x)^2.</p>
	 */
	public static Activation tanH()
	{
		return new Activation() {
			@Override
			public float[] f(float[] x) {
				float[] A = new float[x.length];
				IntStream.range(0, x.length).forEach(
						a -> A[a] = Utils.tanH(x[a]));
				return A;			
			}
			@Override
			public float[] der(float[] x) {
				float[] A = new float[x.length];
				IntStream.range(0, x.length).forEach(
						a -> A[a] = (float) (1-Math.pow(Utils.tanH(x[a]), 2)));
				return A;	
			}
		};
	}
	/** <p>The Softmax ativation function, given by f(x)_i = exp(x_i)/SUM(exp(x_j)|j=1).</p>
	 * <p>The derivative of this function is f'(x)_i = f(x)_i(&#x3B4_ij - f(x)_j).</p>
	 */
	public static Activation softMax()
	{
		return new Activation () {
			@Override
			public float[] f(float[] x) {
				float[] exp = Utils.exps(x), A = new float[x.length];
				float sum = Utils.sum(exp);
				for(int a = 0; a < x.length; a++)
					A[a] = exp[a]/sum;
				return A;
			}
			@Override
			public float[] der(float[] x) {
				float[] f = f(x), A = new float[x.length];
				for(int i = 0; i < x.length; i++)
					for(int j = 0; j < x.length; j++)
						A[i] = f[i]*((i == j ? 1 : 0) - f[j]);
				return A;
			}
		};
	}
	/** <p>The softplus ativation function, given by f(x) = ln(1+exp(x)).</p>
	 * <p>The derivative of this function is 1/(1+exp(-x)).</p>
	 */
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