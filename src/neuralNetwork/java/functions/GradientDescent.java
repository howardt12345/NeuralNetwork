package neuralNetwork.java.functions;

import neuralNetwork.java.layer.Connection;
import neuralNetwork.java.layer.Layer;
import neuralNetwork.java.utils.Matrix;
import neuralNetwork.java.utils.Utils;

public abstract class GradientDescent {
	public abstract void update(Connection c, float eta, int size);
	public abstract void update(Layer l, float[] nebla_b, float eta, int size);

	public static GradientDescent SGD()
	{
		return new GradientDescent() {
			@Override
			public void update(Connection c, float eta, int size) {
				c.setWeights(Matrix.subtract(c.getWeights(), Matrix.scalarMultiply(c.getDeltaWeights(), eta/size)));
			}
			@Override
			public void update(Layer l, float[] nebla_b, float eta, int size) {
				l.setBias(Utils.subtract(l.biases(), Utils.multiply(nebla_b, eta/size)));
			}
		};
	}
}