package neuralNetwork.java.functions;

import java.util.stream.*;

import neuralNetwork.java.utils.*;

public abstract class Activation {
	private Activation() {}
	public abstract Matrix f(Matrix x);
	public abstract Matrix der(Matrix x);
	/** <p>The Identity activation function, given by f(x) = x.</p>
	 * <p>The derivative of this function is f'(x) = 1.</p>
	 */
	public static Activation identity()
	{
		return new Activation() {
			@Override
			public Matrix f(Matrix x) {
				return x;
			}
			@Override
			public Matrix der(Matrix x) {
				return Matrix.ofValue(1, x.rows(), x.columns());
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
			public Matrix f(Matrix x) {
				Matrix A = new Matrix(x.rows(), x.columns());
				IntStream.range(0, x.length()).forEach(
						a -> A.set(x.get(a) < 0 ? 0 : 1, a));
				return A;
			}
			@Override
			public Matrix der(Matrix x) {
				Matrix A = new Matrix(x.rows(), x.columns());
				IntStream.range(0, x.length()).forEach(
						a -> A.set(x.get(a) != 0 ? 0 : null, a));
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
			public Matrix f(Matrix x) {
				Matrix A = new Matrix(x.rows(), x.columns());
				IntStream.range(0, x.length()).forEach(
						a -> A.set(Utils.sigmoid(x.get(a)), a));
				return A;			
			}
			@Override
			public Matrix der(Matrix x) {
				Matrix A = new Matrix(x.rows(), x.columns()), s = f(x);
				IntStream.range(0, x.length()).forEach(
						a -> A.set(s.get(a)*(1-s.get(a)), a));
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
			public Matrix f(Matrix x) {
				Matrix A = new Matrix(x.rows(), x.columns());
				IntStream.range(0, x.length()).forEach(
						a -> A.set(x.get(a) < 0 ? 0 : x.get(a), a));
				return A;
			}
			@Override
			public Matrix der(Matrix x) {
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
			public Matrix f(Matrix x) {
				Matrix A = new Matrix(x.rows(), x.columns());
				IntStream.range(0, x.length()).forEach(
						a -> A.set(x.get(a) < 0 ? 0.01f*x.get(a) : x.get(a), a));
				return A;			
			}
			@Override
			public Matrix der(Matrix x) {
				Matrix A = new Matrix(x.rows(), x.columns());
				IntStream.range(0, x.length()).forEach(
						a -> A.set(x.get(a) < 0 ? 0.01f : 1, a));
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
			public Matrix f(Matrix x) {
				Matrix A = new Matrix(x.rows(), x.columns());
				IntStream.range(0, x.length()).forEach(
						i -> A.set(x.get(i) < 0 ? a*x.get(i) : x.get(i), i));
				return A;			
			}
			@Override
			public Matrix der(Matrix x) {
				Matrix A = new Matrix(x.rows(), x.columns());
				IntStream.range(0, x.length()).forEach(
						i -> A.set(x.get(i) < 0 ? a : 1, i));
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
			public Matrix f(Matrix x) {
				Matrix A = new Matrix(x.rows(), x.columns());
				IntStream.range(0, x.length()).forEach(
						a -> A.set(Utils.tanH(x.get(a)), a));
				return A;			
			}
			@Override
			public Matrix der(Matrix x) {
				Matrix A = new Matrix(x.rows(), x.columns());
				IntStream.range(0, x.length()).forEach(
						a -> A.set((float) (1-Math.pow(Utils.tanH(x.get(a)), 2)), a));
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
			public Matrix f(Matrix x) {
				Matrix exp = Matrix.exps(x), A = new Matrix(x.rows(), x.columns());
				float sum = exp.sum();
				for(int a = 0; a < x.length(); a++)
					A.set(exp.get(a)/sum, a);
				return A;
			}
			@Override
			public Matrix der(Matrix x) {
				Matrix f = f(x), 
						A = new Matrix(x.rows(), x.columns()), 
						k = Matrix.identity(x.rows(), x.columns());
				for(int i = 0; i < x.length(); i++)
					for(int j = 0; j < x.length(); j++)
						A.set(f.get(i)*k.get(i, j) - f.get(j), i);
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
			public Matrix f(Matrix x) {
				Matrix A = new Matrix(x.rows(), x.columns());
				IntStream.range(0, x.length()).forEach(
						a -> A.set((float) Math.log(1+Utils.exp(x.get(a))), a));
				return A;
			}
			public Matrix der(Matrix x) {
				return sigmoid().f(x);
			}
		};
	}
}