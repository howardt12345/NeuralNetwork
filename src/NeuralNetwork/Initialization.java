package NeuralNetwork;

import java.util.stream.*;

public abstract class Initialization {
	private Initialization() {}
	public abstract float f(float x, float y);
	public abstract void initialize(Connection c);
	public static float random(float x, float y)
	{
		return x + (float)Math.random() * (y-x);
	}
	public static float xavier(float x, float y)
	{
		return random(0, 2/(x+y));
	}
	public static float glorotNormal(float x, float y)
	{
		return random(0, (float) Math.sqrt(xavier(x, y)));
	}
	public static float glorotUniform(float x, float y)
	{
		return random((float) -Math.sqrt(6/(x+y)), (float) Math.sqrt(6/(x+y)));
	}
	public static float He(float x, float y)
	{
		return random(0, (float) (y*Math.sqrt(1/x)));
	}
	public static float HeNormal(float x, float y)
	{
		return random((float) (Math.sqrt(6/x)), (float) (Math.sqrt(6/x)));
	}
	public static Initialization zeros()
	{
		return new Initialization() {
			
			public float f(float x, float y) {
				return 0;
			}
			
			public void initialize(Connection c) {
				IntStream.range(0, c.size()).forEach(a -> c.weights.matrix[a] = f(0, 0));
			}
		};
	}
	public static Initialization ones()
	{
		return new Initialization() {
			
			public float f(float x, float y) {
				return 1;
			}
			
			public void initialize(Connection c) {
				IntStream.range(0, c.size()).forEach(a -> c.weights.matrix[a] = f(0, 0));
			}
		};
	}
	public static Initialization constant(float x)
	{
		return new Initialization() {
			
			public float f(float x, float y) {
				return x;
			}
			
			public void initialize(Connection c) {
				IntStream.range(0, c.size()).forEach(a -> c.weights.matrix[a] = f(x, 0));
			}
		};
	}
	public static Initialization randomNormal()
	{
		return new Initialization() {
			
			public float f(float x, float y) {
				return random(0, 1);
			}
			
			public void initialize(Connection c) {
				IntStream.range(0, c.size()).forEach(a -> c.weights.matrix[a] = f(c.getInput().size(), c.getOutput().size()));
			}
		};
	}
	public static Initialization randomUniform()
	{
		return new Initialization() {
			
			public float f(float x, float y) {
				return random(-1, 1);
			}
			
			public void initialize(Connection c) {
				IntStream.range(0, c.size()).forEach(a -> c.weights.matrix[a] = f(c.getInput().size(), c.getOutput().size()));
			}
		};
	}
	public static Initialization Random(float x, float y)
	{
		return new Initialization() {
			
			public float f(float x, float y) {
				return random(x, y);
			}
			
			public void initialize(Connection c) {
				IntStream.range(0, c.size()).forEach(a -> c.weights.matrix[a] = f(c.getInput().size(), c.getOutput().size()));
			}
		};
	}
	public static Initialization xavier()
	{
		return new Initialization() {
			
			public float f(float x, float y) {
				return xavier(x, y);
			}
			
			public void initialize(Connection c) {
				IntStream.range(0, c.size()).forEach(a -> c.weights.matrix[a] = f(c.getInput().size(), c.getOutput().size()));
			}
		};
	}
	public static Initialization glorotNormal()
	{
		return new Initialization() {
			
			public float f(float x, float y) {
				return glorotNormal(x, y);
			}
			
			public void initialize(Connection c) {
				IntStream.range(0, c.size()).forEach(a -> c.weights.matrix[a] = f(c.getInput().size(), c.getOutput().size()));
			}
		};
	}
	public static Initialization glorotUniform()
	{
		return new Initialization() {
			
			public float f(float x, float y) {
				return glorotUniform(x, y);
			}
			
			public void initialize(Connection c) {
				IntStream.range(0, c.size()).forEach(a -> c.weights.matrix[a] = f(c.getInput().size(), c.getOutput().size()));
			}
		};
	}
	public static Initialization HeSigmoid()
	{
		return new Initialization() {
			
			public float f(float x, float y) {
				return He(x, y);
			}
			
			public void initialize(Connection c) {
				IntStream.range(0, c.size()).forEach(a -> c.weights.matrix[a] = f(c.getInput().size(), 1));
			}
		};
	}
	public static Initialization HeReLu()
	{
		return new Initialization() {
			
			public float f(float x, float y) {
				return He(x, y);
			}
			
			public void initialize(Connection c) {
				IntStream.range(0, c.size()).forEach(a -> c.weights.matrix[a] = f(c.getInput().size(), (float) Math.sqrt(2)));
			}
		};
	}
	public static Initialization HeLeakyReLu()
	{
		return new Initialization() {
			
			public float f(float x, float y) {
				return He(x, y);
			}
			
			public void initialize(Connection c) {
				IntStream.range(0, c.size()).forEach(a -> c.weights.matrix[a] = f(c.getInput().size(), (float) Math.sqrt(2/(1+0.01*2))));
			}
		};
	}
	public static Initialization HeNormal()
	{
		return new Initialization() {
			
			public float f(float x, float y) {
				return HeNormal(x, y);
			}
			
			public void initialize(Connection c) {
				IntStream.range(0, c.size()).forEach(a -> c.weights.matrix[a] = f(c.getInput().size(), 0));
			}
		};
	}
}
