package Java.neuralNetwork;

public abstract class Cost {
	public abstract float f(float[] A, float[] E);
	public abstract float[] delta(float[] A, float[] E, Layer l);
	private Cost () {}
	
	public static Cost quadratic()
	{
		return new Cost() {
			
			public float f(float[] A, float[] E) {
				assert(A.length == E.length);
				float tmp = 0;
				for(int a = 0; a < A.length; a++)
					tmp += 0.5f*Math.pow(A[a] - E[a], 2);
				return tmp;
			}
			public float[] delta(float[] A, float[] E, Layer l) {
				return Utils.multiply(Utils.subtract(A, E), l.derivatives());
			}
		};
	}
}
