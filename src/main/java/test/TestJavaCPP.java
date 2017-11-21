package main.java.test;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

@Platform(include = "test/test.h")
public class TestJavaCPP {
	static 
	{
		Loader.load();
	}
	public native void sayhello();
	
	public static void main(String[] args)
	{
		
	}
}
