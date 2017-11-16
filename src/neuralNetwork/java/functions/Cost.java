package neuralNetwork.java.functions;

import neuralNetwork.java.utils.*;

public abstract class Cost {
	public abstract float f(Matrix A, Matrix E);
	public abstract Matrix delta(Matrix A, Matrix E, Matrix derivatives);
	private Cost () {}
	
	public static Cost mst()
	{
		return new Cost() {
			@Override
			public float f(Matrix A, Matrix E) 
			{
				float tmp = 0;
				for(int a = 0; a < A.length(); a++)
					tmp += Math.pow(A.get(a) - E.get(a), 2);
				return 0.5f*tmp;
			}
			@Override
			public Matrix delta(Matrix A, Matrix E, Matrix derivatives) 
			{
				return Matrix.haramardProduct(Matrix.subtract(A, E), derivatives);
			}
		};
	}
	public static Cost crossEntropy()
	{
		return new Cost() {
			@Override
			public float f(Matrix A, Matrix E) 
			{
				float tmp = 0;
				for(int a = 0; a < A.length(); a++)
					tmp += -(E.get(a) * Utils.ln(A.get(a))
							+ (1-E.get(a))*Utils.ln(1 - A.get(a)));
				return tmp;
			}
			@Override
			public Matrix delta(Matrix A, Matrix E, Matrix derivatives) 
			{
/*				assert(A.length == E.length) : A.length + " != " + E.length;
				Matrix tmp = new float[A.length];
				for(int a = 0; a < tmp.length; a++)
					tmp[a] = (A[a] - E[a])/((1 - A[a])*A[a]);
				return tmp;*/
				return Matrix.subtract(A, E);
			}
		};
	}
}
