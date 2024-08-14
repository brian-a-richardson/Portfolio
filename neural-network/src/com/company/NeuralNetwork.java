package com.company;

import java.util.ArrayList;
import java.util.Arrays;

public class NeuralNetwork {
    ArrayList<ArrayList<NetworkNode>> nodes;
    ArrayList<ArrayList<Edge>> edges;


    NeuralNetwork(int inputNodes, int outputNodes, int hiddenLayers, int hiddenNodes){
        // initialize the array
        nodes = new ArrayList<>();
        edges = new ArrayList<>();
        ArrayList<NetworkNode> tempNodes = new ArrayList<>();

        // create input nodes
        for (int i = 0; i < inputNodes; i++){
            tempNodes.add(new NetworkNode(0.0));
        }

        nodes.add(tempNodes);

        // create hidden layers
        for (int i = 0; i < hiddenLayers; i++){
            tempNodes = new ArrayList<>();
            for (int j = 0; j < hiddenNodes; j++) {
                tempNodes.add(new NetworkNode(0.0));
            }
            nodes.add(tempNodes);
        }

        tempNodes = new ArrayList<>();

        // create output nodes
        for (int i = 0; i < outputNodes; i++){
            tempNodes.add(new NetworkNode(0.0));
        }

        nodes.add(tempNodes);

        for (int i = 0; i < nodes.size() - 1; i++){
            edges.add(generateDefaultEdges(nodes.get(i), nodes.get(i+1)));
        }

    }
        /*
            Methods
        */
    public void debugPrint(){
        for (ArrayList<NetworkNode> a: nodes){
            System.out.println(a.size());
        }

        for (ArrayList<Edge> a: edges){
            System.out.println(a.size());
        }

    }

    private ArrayList<Edge> generateDefaultEdges(ArrayList<NetworkNode> leftNetworkNodes, ArrayList<NetworkNode> rightNetworkNodes){
        ArrayList<Edge> edges = new ArrayList<Edge>();

        // generate edges between nodes
        for (NetworkNode a: leftNetworkNodes){
            for (NetworkNode b: rightNetworkNodes) {
                edges.add(new Edge(0, a, b));
            }
        }
        return edges;
    }

    public ArrayList<Double> run(ArrayList<Double> input){

        for (int i = 0; i < input.size(); ++i){
            nodes.get(0).get(i).value = input.get(i);
        }

        //the activation of the  neuron at layer x, neuron y is equal to
        //the sum of each neuron at layer x-1 times the weight between that neuron and neuron y
        //then you take that sum, add node y's bias, and then apply the activation function
        //and assign to the activation of node y
        // a[x,y] = F(sum[n=0 to neuronsInALayer](a[x-1, n] * w[x-1,n;x,y]) + b[x,y])
        for (int i = 0; i < input.size(); ++i){

        }


        return new ArrayList<Double>();
    }

}

