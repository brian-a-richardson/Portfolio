package com.company;

import java.util.ArrayList;

public class Edge {
    double weight;
    NetworkNode leftNode;
    NetworkNode rightNode;

    Edge(double weight, NetworkNode leftNode, NetworkNode rightNode){
        this.weight = weight;
        this.leftNode = leftNode;
        this.rightNode = rightNode;
    }
}
