package neuralNetwork.java.networks;

import java.util.*;
import java.io.*;

import neuralNetwork.java.layer.*;
import neuralNetwork.java.utils.Matrix;

public abstract class NeuralNetwork {
	protected Layer[] layers;
	protected Connection[] weights;
}