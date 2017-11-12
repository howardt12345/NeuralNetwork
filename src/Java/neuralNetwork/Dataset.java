package Java.neuralNetwork;

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