/**
File Name: BenchmarkSort.java
Author(s): Brian Richardson
Date: 7/13/2021
Description:  Main method for benchmarking heap sort
 */


public class BenchmarkSorts{
    static {
        long start = System.nanoTime();
        ManualClassLoader.load();
        long end = System.nanoTime();
        System.out.println("Warm Up Time : " + (end - start));
    }
    public static void main(String[] args){
        long start = System.nanoTime();
        ManualClassLoader.load();
        long end = System.nanoTime();
        System.out.println("Total Time taken :" + (end - start));

        YourSort yourSort = new YourSort(10, 100);

    }
}

/**
 Java Warm Up Classes
 Code Source:  https://en.getdocs.org/java-jvm-warmup/
 */
class ManualClassLoader {
    protected static void load(){
        for (int i = 0; i < 100000; i++){
            Dummy dummy = new Dummy();
            dummy.m();
        }
    }
}

class Dummy {
    public void m(){

    }
}