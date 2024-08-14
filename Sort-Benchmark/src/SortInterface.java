/**
File Name: BenchmarkSort.java
Author(s): Brian Richardson
Date: 7/13/2021
Description:  Main method for benchmarking heap sort
 */

public interface SortInterface {
    /**
    Recursive HEAP Sort
    Source:
    GeeksforGeeks. (2019, March 27). Java Program for Heap Sort. GeeksforGeeks.
    https://www.geeksforgeeks.org/java-program-for-heap-sort/.
     */
    default int recursiveSort(int arr[])
    {
        int n = arr.length;
        int count = 0; // counter for benchmarking

        // Build heap (rearrange array)
        for (int i = n / 2 - 1; i >= 0; i--)
            count += heapify(arr, n, i);

        // One by one extract an element from heap
        for (int i=n-1; i>=0; i--)
        {
            // Move current root to end
            int temp = arr[0];
            arr[0] = arr[i];
            arr[i] = temp;

            // call max heapify on the reduced heap
            count += heapify(arr, i, 0);
        }
        return count;
    }

    // To heapify a subtree rooted with node i which is
    // an index in arr[]. n is size of heap
    default int heapify(int arr[], int n, int i)
    {
        int count = 0; // counter for benchmarking
        int largest = i;  // Initialize largest as root
        int l = 2*i + 1;  // left = 2*i + 1
        int r = 2*i + 2;  // right = 2*i + 2

        count += 1;

        // If left child is larger than root
        if (l < n && arr[l] > arr[largest])
            largest = l;
            count += 1;

        // If right child is larger than largest so far
        if (r < n && arr[r] > arr[largest])
            largest = r;
            count += 1;

        // If largest is not root
        if (largest != i)
        {
            int swap = arr[i];
            arr[i] = arr[largest];
            arr[largest] = swap;

            // Recursively heapify the affected sub-tree
            count += heapify(arr, n, largest);
        }
        return count;
    }

    /**
    Iterative HEAP sort
    Source:
    GeeksforGeeks. (2019, July 31). Iterative HeapSort. GeeksforGeeks.
    https://www.geeksforgeeks.org/iterative-heap-sort/.

     */

    // function build Max Heap where value
    // of each child is always smaller
    // than value of their parent
    default int buildMaxHeap(int arr[], int n)
    {
        int count = 0; // counter for benchmarking
        for (int i = 1; i < n; i++)
        {
            // if child is bigger than parent
            if (arr[i] > arr[(i - 1) / 2])
            {
                int j = i;
                count += 1;

                // swap child and parent until
                // parent is smaller
                while (arr[j] > arr[(j - 1) / 2])
                {
                    swap(arr, j, (j - 1) / 2);
                    count += 1;
                    j = (j - 1) / 2;
                }
            }
        }
        return count;
    }

    default int iterativeSort(int arr[], int n)
    {
        int count = 0; // counter for benchmarking
        count = buildMaxHeap(arr, n);

        for (int i = n - 1; i > 0; i--)
        {
            // swap value of first indexed
            // with last indexed
            swap(arr, 0, i);
            count += 1;

            // maintaining heap property
            // after each swapping
            int j = 0, index;

            do
            {
                index = (2 * j + 1);

                // if left child is smaller than
                // right child point index variable
                // to right child
                if (index < (i - 1) && arr[index] < arr[index + 1])
                    index++;
                    count+=1;

                // if parent is smaller than child
                // then swapping parent with child
                // having higher value
                if (index < i && arr[j] < arr[index])
                    swap(arr, j, index);
                    count+=1;

                j = index;
                count += 1;

            } while (index < i);
        }
        return count;
    }

    static void swap(int[] a, int i, int j) {
        int temp = a[i];
        a[i]=a[j];
        a[j] = temp;
    }

    /**
    Utility Functions
     */

    default int getCount(){
        int count = 0;

        return count;
    }

    default long getTime(){
        long time = System.nanoTime();

        return time;
    }

}
