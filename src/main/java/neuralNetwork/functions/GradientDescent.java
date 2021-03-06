package main.java.neuralNetwork.functions;

import main.java.neuralNetwork.layer.*;
import main.java.neuralNetwork.utils.*;

public abstract class GradientDescent {
	public abstract void update(Connection c, float eta, int size);
	public abstract void update(Layer l, float eta, int size);
	
	
	public static GradientDescent SGD()
	{
		return new GradientDescent() {
			@Override
			public void update(Connection c, float eta, int size) {
				c.setWeights(Matrix.subtract(c.getWeights(), Matrix.multiply(c.getDeltaWeights(), eta/size)));
			}
			@Override
			public void update(Layer l, float eta, int size) {
				l.setBias(Matrix.subtract(l.bias(), Matrix.multiply(l.nebla_b(), eta/size)));
			}
		};
	}
}