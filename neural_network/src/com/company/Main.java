package com.company;

public class Main {

    public static void main(String[] args) {
	    System.out.println("Generating neural network");
	    // generate neural network
        NeuralNetwork myBrain = new NeuralNetwork(8, 1, 6, 8);
        myBrain.debugPrint();

    }
}
