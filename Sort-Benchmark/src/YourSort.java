import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

/**
File Name: YourSort.java
Author(s): Brian Richardson
Date: 7/13/2021
Description:  Class for sorting and benchmarking arrays
 */

public class YourSort implements SortInterface{
    // attributes
    ArrayList<int[]> arrays = new ArrayList<>();
    ArrayList<Integer> sizes = new ArrayList<>();
    ArrayList<Long> avgIterativeCount = new ArrayList<>();
    ArrayList<Double> coefIterativeCount = new ArrayList<>();
    ArrayList<Long> avgIterativeTime = new ArrayList<>();
    ArrayList<Double> coefIterativeTime = new ArrayList<>();
    ArrayList<Long> avgRecursiveCount = new ArrayList<>();
    ArrayList<Double> coefRecursiveCount = new ArrayList<>();
    ArrayList<Long> avgRecursiveTime = new ArrayList<>();
    ArrayList<Double> coefRecursiveTime = new ArrayList<>();
    Random random = new Random();

    // Constructor
    YourSort(int numberOfArrays, int elements){
        sortAll(numberOfArrays, elements);
    }

    /*
    Methods
     */

    // create random array of n elements
    private int[] createRandomArray(int elements){
        int[] randomArray = new int[elements];

        for(int i = 0; i < elements; i++){
            randomArray[i] = random.nextInt(100);
        }

        return randomArray;
    }

    // sort all arrays
    private void sortAll(int numberOfArrays, int elements){
        long startTime;
        long endTime;
        long time;
        int count;

        for(int i = 1; i < numberOfArrays + 1; i++){
            // for each size perform benchmark procedure
            int size = i * elements;
            sizes.add(size);

            // store data sets for computation
            long[] iterativeTimes = new long[50];
            long[] recurisveTimes = new long[50];
            long[] iterativeCount = new long[50];
            long[] recursiveCount = new long[50];


            for(int j = 0; j < 50; j++) {
                // create a random array and sort it 50 times

                int[] testArray = createRandomArray(size);

                // iterative sort the array
                startTime = getTime();
                count = iterativeSort(testArray, size);
                endTime = getTime();
                time = endTime - startTime;

                System.out.println("Array " + i + ", Iterative Iteration " + j + " = " + time + " - " + count);
                iterativeCount[j] = count;
                iterativeTimes[j] = time;

                // recursive sort the array
                startTime = getTime();
                count = recursiveSort(testArray);
                endTime = getTime();
                time = endTime - startTime;

                System.out.println("Array " + i + ", Recursive Iteration " + j + " = " + time + " - " + count);
                recursiveCount[j] = count;
                recurisveTimes[j] = time;
            }

            // calculate and store the iterative data
            avgIterativeCount.add(getMean(iterativeCount));
            coefIterativeCount.add(getCoefficient(iterativeCount));
            avgIterativeTime.add(getMean(iterativeTimes));
            coefIterativeTime.add(getCoefficient(iterativeTimes));

            // calculate and store recursive data
            avgRecursiveCount.add(getMean(recursiveCount));
            coefRecursiveCount.add(getCoefficient(recursiveCount));
            avgRecursiveTime.add(getMean(recurisveTimes));
            coefRecursiveTime.add(getCoefficient(recurisveTimes));

        }

        createDataFiles();
    }

    // support math functions

    private long getMean(long[] arr){
        long average;
        long sum = 0;
        for(int i = 0; i < arr.length; i++){
            sum += arr[i];
        }
        average = sum / arr.length;
        return average;
    }

    private double getMean(double[] arr){
        double average;
        double sum = 0;
        for(int i = 0; i < arr.length; i++){
            sum += arr[i];
        }
        average = sum / arr.length;
        return average;
    }

    private double getCoefficient(long[] arr){
        double mean = getMean(arr);
        double coefficient;
        double stdDev;
        double[] updatedVaues = new double[arr.length];

        for(int i = 0; i < updatedVaues.length; i++){
            updatedVaues[i] = Math.pow((arr[i] - mean), 2);
        }

        double updatedMean =  getMean(updatedVaues);
        stdDev = Math.sqrt(updatedMean);
        coefficient = (stdDev / mean) * 100;

        return coefficient;
    }

    // arrays to file

    private void createDataFiles(){
        try {
            // create iterative data file
            File iterative = new File("Iterative.txt");
            if(iterative.createNewFile()){
                System.out.println("File created: " + iterative.getName());
            } else {
                System.out.println("File already exists.");
            }

            // create a file writer
            FileWriter writer = new FileWriter("Iterative.txt");

            // build a comma separated string from arrays and write the string to file
            String line;

            for(int i = 0; i < sizes.size(); i++){
                line = sizes.get(i) + "," + avgIterativeCount.get(i) + "," + Math.round(coefIterativeCount.get(i) * 100) + "," + avgIterativeTime.get(i) + "," + Math.round(coefIterativeTime.get(i) * 100) + "\n";
                System.out.println(line);
                writer.write(line);
            }
            writer.close();

        } catch (IOException e){
            e.printStackTrace();
        }

        try {
            // create recursive data file
            File recursive = new File("Recursive.txt");
            if(recursive.createNewFile()){
                System.out.println("File created: " + recursive.getName());
            } else {
                System.out.println("File already exists.");
            }

            // create a file writer
            FileWriter writer = new FileWriter("Recursive.txt");

            // build a comma separated string from arrays and write the string to file
            String line;

            for(int i = 0; i < sizes.size(); i++){
                line = sizes.get(i) + "," + avgRecursiveCount.get(i) + "," + Math.round(coefRecursiveCount.get(i) * 100) + "," + avgRecursiveTime.get(i) + "," + Math.round(coefRecursiveTime.get(i) * 100) + "\n";
                System.out.println(line);
                writer.write(line);
            }
            writer.close();

        } catch (IOException e){
            e.printStackTrace();
        }

    }
}
