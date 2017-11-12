package Java.neuralNetwork.functions;

import Java.neuralNetwork.Utils;

public abstract class Cost {
	public abstract float f(float[] A, float[] E);
	public abstract float[] delta(float[] A, float[] E);
	private Cost () {}
	
	public static Cost mst()
	{
		return new Cost() {
			
			public float f(float[] A, float[] E) {
				assert(A.length == E.length);
				float tmp = 0;
				for(int a = 0; a < A.length; a++)
					tmp += Math.pow(A[a] - E[a], 2);
				return 0.5f*tmp;
			}
			public float[] delta(float[] A, float[] E) {
				return Utils.subtract(A, E);
			}
		};
	}
}
