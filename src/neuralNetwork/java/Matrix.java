package neuralNetwork.java;

import java.io.*;
import java.nio.file.Files;
import java.util.*;
import java.util.stream.*;
import com.aparapi.*;

/** The Matrix class.*/
public class Matrix {
	/** The Matrix.*/
	protected float[] matrix = 
		{
			1, 0, 0, 0,
			0, 1, 0, 0, 
			0, 0, 1, 0,
			0, 0, 0, 1
		};
	private int rows = 4, columns = 4;
	/** New Matrix. Default 4x4 Matrix.*/
	public Matrix () {}
	/** New Matrix from 2D array.
	 * @param matrix the 2D array
	 */
	public Matrix (float[][] matrix) 
	{
		this.matrix = new float[matrix.length * matrix[0].length];
		int k = 0;
		this.rows = matrix.length;
		this.columns = matrix[0].length;
	    for (int i = 0; i < matrix.length; i++) {
	        for (int j = 0; j < matrix[0].length; j++) {
	            this.matrix[k++] = matrix[i][j];
	        }
	    }
	}
	public Matrix (float[] vector)
	{
		this.rows = vector.length;
		this.columns = 1;
		this.matrix = vector;
	}
	/** New Matrix from a String.
	 * @param string the Matrix, represented as a String.
	 */
	public Matrix (String string)
	{
		try
		{
			String[] s = string.split(",");
			rows = Integer.parseInt(s[0].replaceAll("#", ""));
			columns = s[1].contains("#") ? Integer.parseInt(s[1].replaceAll("#", "")) : rows;
			this.matrix = new float[rows*columns];
			for (int a = columns == rows ? 1 : 2; a < s.length; a++) 
			{
				matrix[a-(columns == rows ? 1 : 2)] = Float.parseFloat(s[a]);
			}
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
	}
	/** New Matrix from file.
	 * @param file the Matrix file.
	 */
	public Matrix (File file)
	{
		try 
		{
			String in = new String(Files.readAllBytes(file.toPath()));
			String[] s = in.split(",");
			rows = Integer.parseInt(s[0].replaceAll("#", ""));
			columns = s[1].contains("#") ? Integer.parseInt(s[1].replaceAll("#", "")) : rows;
			this.matrix = new float[rows*columns];
			for (int a = columns == rows ? 1 : 2; a < s.length; a++) 
			{
				matrix[a-(columns == rows ? 1 : 2)] = Float.parseFloat(s[a]);
			}
		} catch (Exception e1) {
			try
			{
				BufferedReader in = new BufferedReader(new FileReader (file));
				String line = in.readLine();
				ArrayList<ArrayList<Float>> tmp = new ArrayList<ArrayList<Float>>();
				while (line != null)
				{
					String[] tokens = line.split(" ");
					ArrayList<Float> tmp1 = new ArrayList<Float>();
					for (int a = 0; a < tokens.length; a++)
					{
						tmp1.add(Float.parseFloat(tokens[a]));
					}
					tmp.add(tmp1);
					line = in.readLine();
				}
				in.close();
				this.matrix = new float[tmp.size() * tmp.get(0).size()];
				rows = tmp.size(); 
				columns = tmp.get(0).size();
				int i = 0;
				for (int a = 0; a < tmp.size(); a++)
				{
					for (int b = 0; b < tmp.get(a).size(); b++)
					{
						matrix[i] = tmp.get(a).get(b);
						i++;
					}
				}
			} catch (Exception e2) {
				e1.printStackTrace();
				e2.printStackTrace();
			}
		}
	}
	/** New Matrix from rows and columns.
	 * @param rows the number of rows.
	 * @param columns the number of columns.
	 */
	public Matrix (int rows, int columns) 
	{
		this.matrix = new float[rows * columns];
		this.rows = rows;
		this.columns = columns;
		int k = 0;
	    for (int i = 0; i < rows; i++) {
	        for (int j = 0; j < columns; j++) {
	            this.matrix[k++] = 0;
	        }
	    }
	}
	public Matrix (int rows, int columns, float[] matrix)
	{
		assert(rows*columns == matrix.length) : rows + " * " + columns + " != " + matrix.length;
		this.rows = rows;
		this.columns = columns;
		this.matrix = matrix;
	}
	/** Identity Matrix. */
	public static Matrix identity () 
	{
		Matrix m = new Matrix ();
		for (int a = 0; a < m.rows(); a++) 
		{
			for (int b = 0; b < m.columns(); b++) 
				m.set (a == b ? 1 : 0, a, b);
		}
		return m;
	}
	public static Matrix diagonal(float[] values)
	{
		Matrix m = new Matrix(values.length, values.length);
		int i = 0;
		for (int a = 0; a < m.rows(); a++) 
		{
			for (int b = 0; b < m.columns(); b++) 
				m.set (a == b ? values[i++] : 0, a, b);
		}
		return m;
	}
	/** Identity Matrix from rows and columns.
	 * @param rows
	 * @param columns
	 */
	public static Matrix identity (int rows, int columns) 
	{
		Matrix m = new Matrix (rows, columns);
		for (int a = 0; a < m.rows(); a++) 
		{
			for (int b = 0; b < m.columns(); b++) 
				m.set (a == b ? 1 : 0, a, b);
		}
		return m;
	}
	/** Overrides values of Matrix with identity Matrix.*/
	public void toIdentity () 
	{
		for (int a = 0; a < rows(); a++) 
		{
			for (int b = 0; b < columns(); b++) 
				set (a == b ? 1 : 0, a, b);
		}
	}
	/** Matrix with all values at 0.*/
	public static Matrix zero () 
	{
		Matrix m = new Matrix ();
		for (int a = 0; a < m.rows(); a++) {
			for (int b = 0; b < m.columns(); b++) 
				m.set(0, a, b);
		}
		return m;
	}
	/** Matrix with all values at 0 from rows and columns.
	 * @param rows the rows
	 * @param columns the columns
	 */
	public static Matrix zero (int rows, int columns) 
	{
		Matrix m = new Matrix (rows, columns);
		for (int a = 0; a < rows; a++) 
		{
			for (int b = 0; b < columns; b++) 
				m.set(0, a, b);
		}
		return m;
	}
	/** Overrides values of Matrix with Matrix with all values at 0.*/
	public void toZero () 
	{
		for (int a = 0; a < rows(); a++) 
		{
			for (int b = 0; b < columns(); b++) 
				set(0, a, b);
		}
	}
	public void randomize()
	{
		IntStream.range(0, matrix.length).forEach(a -> matrix[a] = (float) Math.random()-0.5f);
	}
	public void randomize(long seed)
	{
		Random r = new Random(seed);
		IntStream.range(0, matrix.length).forEach(a -> matrix[a] = (float) r.nextFloat()-0.5f);
	}
	public static Matrix add(Matrix m1, Matrix m2)
	{
		assert(m1.rows == m2.rows && m1.columns == m2.columns)
			: m1.rows + " != " + m2.rows + " & " + m1.columns + " != " + m2.columns;
		Matrix m = new Matrix(m1.rows, m1.columns);
		for(int a = 0; a < m1.matrix.length; a++)
		{
			m.matrix[a] = m1.matrix[a] + m2.matrix[a];
		}
		return m;
	}
	public static Matrix subtract(Matrix m1, Matrix m2)
	{
		assert(m1.rows == m2.rows && m1.columns == m2.columns)
			: m1.rows + " != " + m2.rows + " & " + m1.columns + " != " + m2.columns;
		Matrix m = new Matrix(m1.rows, m1.columns);
		for(int a = 0; a < m1.matrix.length; a++)
		{
			m.matrix[a] = m1.matrix[a] - m2.matrix[a];
		}
		return m;
	}
	public static Matrix subtract(Matrix m1, float x)
	{
		Matrix m = new Matrix(m1.rows, m1.columns);
		for(int a = 0; a < m1.matrix.length; a++)
		{
			m.matrix[a] = m1.matrix[a] - x;
		}
		return m;
	}
	public static Matrix scalarMultiply(Matrix m1, float scalar)
	{
		Matrix m = new Matrix(m1.rows, m1.columns);
		for(int a = 0; a < m.matrix.length; a++)
		{
			m.matrix[a] = m1.matrix[a] * scalar;
		}
		return m;
	}
	/** Multiplies two Matrices.
	 * @param m1 the first Matrix
	 * @param m2 the second Matrix
	 */
	public static Matrix multiply (Matrix m1, Matrix m2) 
	{
		assert(m1.columns() == m2.rows()) : m1.columns + " != " + m2.rows;
		Matrix m = new Matrix (m1.rows(), m2.columns());
		for (int b = 0; b < m2.columns(); b++) 
		{
			for (int a = 0; a < m1.rows(); a++)
			{
				float sum = 0;
				for (int c = 0; c < m2.rows(); c++)
				{
					float elementA = m1.matrix[a*m1.columns + c];
					float elementB = m2.matrix[c*m2.columns + b];
					sum += elementA * elementB;
				}
				m.set(sum, a, b);
			}
		}
		return m;
	}
	/** Multiplies 2 Matrixes in Parallel.
	 * @param m1 the first Matrix
	 * @param m2 the second Matrix
	 */
	public static Matrix multiplyParallel (Matrix m1, Matrix m2)
	{
		assert(m1.columns() == m2.rows()) : m1.columns + " != " + m2.rows;
		Matrix m = new Matrix (m1.rows(), m2.columns());
		IntStream.range(0, m2.columns()).parallel().forEach((b) -> {
			for (int a = 0; a < m1.rows(); a++)
			{
				float sum = 0;
				for (int c = 0; c < m2.rows(); c++)
				{
					float elementA = m1.matrix[a*m1.columns + c];
					float elementB = m2.matrix[c*m2.columns + b];
					sum += elementA * elementB;					
				}
				m.set(sum, a, b);
			}
		});
		return m;
	}
	/** Multiplies 2 Matrixes with Aparapi.
	 * @param m1 the first Matrix
	 * @param m2 the second Matrix
	 */
	public static Matrix multiplyAparapi (Matrix m1, Matrix m2) 
	{
		assert(m1.columns() == m2.rows()) : m1.columns + " != " + m2.rows;
		Matrix m = new Matrix (m1.rows(), m2.columns());
		Kernel k1 = new Kernel() {
			@Override
			public void run() {
				int a = getGlobalId(0);
				int b = getGlobalId(1);
				float sum = 0;
				for (int c = 0; c < m1.columns; c++)
				{
					float elementA = m1.matrix[a*m1.columns + c];
					float elementB = m2.matrix[c*m2.columns + b];
					sum += elementA * elementB;
				}
				m.matrix[a*m2.columns + b] = sum;
			}
		};
		Range range = Range.create2D(m1.rows(), m2.columns());
		k1.execute(Range.create(1));
        k1.execute(range);
        //System.out.println("Execution mode: " + k1.getExecutionMode());
		return m;
	}
	/** Returns the product of current Matrix and another Matrix.
	 * @param m1 the Matrix
	 */
	public Matrix multiply (Matrix m1) 
	{
		assert(columns() == m1.rows());
		Matrix m = new Matrix (rows(), m1.columns());
		for (int a = 0; a < rows(); a++) 
		{
			for (int b = 0; b < m1.columns(); b++) 
			{
				float sum = 0;
				for (int c = 0; c < m1.rows(); c++) 
					sum += get(a, c)*m1.get(c, b);
				m.set(sum, a, b);
				sum = 0;
			}
		}
		return m;
	}
	public static Matrix haramardProduct(Matrix m1, Matrix m2)
	{
		assert(m1.rows == m2.rows && m1.columns == m2.columns)
		: m1.rows + " != " + m2.rows + " & " + m1.columns + " != " + m2.columns;
		Matrix m = new Matrix(m1.rows, m1.columns);
		for(int a = 0; a < m1.matrix.length; a++)
		{
			m.matrix[a] = m1.matrix[a] * m2.matrix[a];
		}
		return m;
	}
	public static Matrix kroneckerProduct(Matrix m1, Matrix m2)
	{
		Matrix m = new Matrix(m1.rows*m2.rows, m1.columns*m2.columns);
		for(int a = 0; a < m1.rows; a++)
		{
			for(int b = 0; b < m1.columns; b++)
			{
				addTo(Matrix.scalarMultiply(m2, m1.get(a, b)), m, a*m1.rows, b*m1.columns);
			}
		}
		return m;
	}
	private static void addTo(Matrix m1, Matrix m2, int row, int column)
	{
		for(int a = row; a < row+m1.rows; a++)
		{
			for(int b = column; b < column+m1.columns; b++)
			{
				m2.set(m1.get(a-row, b-column), a, b);
			}
		}
	}
	public static Matrix horizontalConcat(Matrix m1, Matrix m2)
	{
		assert(m1.rows == m2.rows) : m1.rows + " != " + m2.rows;
		Matrix m = new Matrix(m1.rows, m1.columns + m2.columns);
		for(int a = 0; a < m.rows; a++)
		{
			for(int b = 0; b < m.columns; b++)
			{
				m.set(b >= m1.columns ? m2.get(a, b-m1.columns) : m1.get(a, b), a, b);
			}
		}
		return m;
	}
	public static Matrix transpose(Matrix m)
	{
		Matrix t = new Matrix(m.columns, m.rows);
		for(int a = 0; a < t.matrix.length; a++)
		{
			t.matrix[a] = m.matrix[(m.columns*(a%m.rows)) + (a/m.rows)];
		}
		return t;
	}
	public Matrix transpose()
	{
		Matrix t = new Matrix(columns, rows);
		for(int a = 0; a < t.matrix.length; a++)
		{
			t.matrix[a] = matrix[(columns*(a%rows)) + (a/rows)];
		}
		return t;
	}
	public static float determinant(Matrix m)
	{
		float result = 0;
		if(m.rows == 1)
			result = m.get(0, 0);
		else
		{
			for(int i = 0; i < m.columns; i++)
			{
				Matrix tmp = new Matrix(m.rows-1, m.columns-1);
				for(int a = 1; a < m.rows; a++)
				{
					for(int b = 0; b < m.columns; b++)
					{
						if (b < i)
							tmp.set(m.get(a, b), a-1, b);
						else
							tmp.set(m.get(a, b), a-1, b-1);
					}
				}
				result += m.get(0, i) * Math.pow(-1, (double) i) * determinant(tmp);
			}
		}
		return result;
	}
	/** Converts this Matrix into String, then writes to file.
	 * @param filename the file name
	 */
	public void convert(String filename)
	{
		try
		{
			FileWriter fw = new FileWriter(filename);
			fw.write(convert());
			fw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	/** Converts this Matrix into a String.*/
	public String convert()
	{
		StringBuilder sb = new StringBuilder();
		sb.append("#" + rows + (columns == rows ? "" : ",#"+columns));
		for (Float f : matrix)
			sb.append(","+f);
		return sb.toString();
	}
	/** Returns true if the 2 Matrices are equal.
	 * @param m the Matrix to compare
	 */
	public boolean equals (Matrix m) 
	{
		for (int a = 0; a < m.rows(); a++) 
		{
			for (int b = 0; b < m.columns(); b++) 
				if (Double.compare(get(a, b), m.get(a, b)) != 0)
					return false;
		}
		return true;
	}
	/** Gets the Matrix value at the provided index.
	 * @param row the row index.
	 * @param column the column index.
	 * @return the Matrix value.
	 */
	public float get (int row, int column) 
	{
		return matrix[row*columns + column];
	}
	/** Sets the Matrix at the provided index.
	 * @param value the value to set.
	 * @param row the row index.
	 * @param column the column index.
	 */
	public void set (float value, int row, int column) 
	{
		matrix[row*columns + column] = value;
	}
	/** Returns the number of rows in the Matrix..*/
	public int rows () 
	{
		return rows;
	}
	/** Returns the number of columns in the Matrix.*/
	public int columns () 
	{
		return columns;
	}
	public float[] matrix()
	{
		return matrix;
	}
	/** Prints the Matrix.*/
	public void print () 
	{
		for (int a = 0; a < rows(); a++)
		{
			for (int b = 0; b < columns(); b++)
				System.out.print(get(a,b) + " ");
			System.out.println();
		}
	}
}