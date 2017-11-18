package neuralNetwork.java.training;

import java.util.*;
import java.io.*;

public class Dataset {
	private Map<float[], float[]> data = new HashMap<float[], float[]>();
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
	public Dataset(File file) throws IOException
	{
		BufferedReader in = new BufferedReader(new FileReader(file));
		String line1, line2;
		while((line1 = in.readLine()) != null && (line2 = in.readLine()) != null)
		{
			String[] rawIn = line1.split(" "), rawTarget = line2.split(" ");
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
	public void toFile(String filename) throws IOException
	{
		FileWriter fw = new FileWriter(new File(filename));
		String s = "";
		data.forEach((x, y) -> {
			try {
				for(int a = 0; a < x.length; a++)
					fw.write(x[a] + " ");
				fw.write("\r\n");
				for(int a = 0; a < y.length; a++)
					fw.write(y[a] + " ");
				fw.write("\r\n");
			}
			catch(IOException e)
			{
				e.printStackTrace();
			}
		});
		fw.close();
	}
}