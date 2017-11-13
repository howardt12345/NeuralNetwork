package Java.neuralNetwork.functions;

import Java.neuralNetwork.Utils;

public abstract class Cost {
	public abstract float f(float[] A, float[] E);
	public abstract float[] delta(float[] A, float[] E);
	private Cost () {}
	
	public static Cost mst()
	{
		return new Cost() {
			@Override
			public float f(float[] A, float[] E) 
			{
				assert(A.length == E.length) : A.length + " != " + E.length;
				float tmp = 0;
				for(int a = 0; a < A.length; a++)
					tmp += Math.pow(A[a] - E[a], 2);
				return 0.5f*tmp;
			}
			@Override
			public float[] delta(float[] A, float[] E) 
			{
				return Utils.subtract(A, E);
			}
		};
	}
	public static Cost crossEntropy()
	{
		return new Cost() {
			@Override
			public float f(float[] A, float[] E) 
			{
				assert(A.length == E.length) : A.length + " != " + E.length;
				float tmp = 0;
				for(int a = 0; a < A.length; a++)
					tmp += -(E[a]  *Utils.ln(A[a]) + (1-E[a])*Utils.ln(1 - A[a]));
				return tmp;
			}
			@Override
			public float[] delta(float[] A, float[] E) 
			{
/*				assert(A.length == E.length) : A.length + " != " + E.length;
				float[] tmp = new float[A.length];
				for(int a = 0; a < tmp.length; a++)
					tmp[a] = (A[a] - E[a])/((1 - A[a])*A[a]);
				return tmp;*/
				return Utils.subtract(A, E);
			}
		};
	}
}
