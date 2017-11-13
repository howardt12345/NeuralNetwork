package neuralNetwork.java.training;

import java.util.*;

public class Dataset {
	Map<float[], float[]> data = new HashMap<float[], float[]>();
	private int size;
	public Dataset(float[][] in, float[][] target)
	{
		assert(in.length == target.length) : in.length + " != " + target.length;
		size = in.length;
		for(int a = 0; a < size; a++)
		{
			data.put(in[a], target[a]);
		}
	}
	public Dataset(String[] in, String[] target)
	{
		assert(in.length == target.length) : in.length + " != " + target.length;
		size = in.length;
		for(int i = 0; i < size; i++)
		{
			String[] rawIn = in[i].split(" "), rawTarget = target[i].split(" ");
			float[] parsedIn = new float[rawIn.length], parsedTarget = new float[rawTarget.length];
			for(int a = 0; a < parsedIn.length; a++)
				parsedIn[a] = Float.parseFloat(rawIn[a]);
			for(int a = 0; a < parsedTarget.length; a++)
				parsedTarget[a] = Float.parseFloat(rawTarget[a]);
			data.put(parsedIn, parsedTarget);
		}
	}
	public List<float[]> getData()
	{
		List<float[]> list = new ArrayList<>(data.keySet());
		Collections.shuffle(list);
		return list;
	}
	public float[] get(float[] in)
	{
		return data.get(in);
	}
	public int size()
	{
		return size;
	}
}