import java.util.ArrayList;
import java.util.Arrays;

public class Process {
    // attributes
    private int id;
    ArrayList<Integer> maxResources;
    ArrayList<Integer> allocatedResources;
    ArrayList<Integer> neededResources = new ArrayList<>();
    boolean complete = false;

    // constructor
    Process(int id, String maxResources, String allocatedResources){
        this.id = id;
        this.maxResources = stringToArray(maxResources);
        this.allocatedResources = stringToArray(allocatedResources);
        updateNeed();
    }

    /*
    Methods
     */

    // get methods
    public int getId(){
        return id;
    }

    public ArrayList<Integer> getNeed(){
        // returns an array list containing the resource needs of this process
        return neededResources;
    }

    public int getAllocated(int index){
        return allocatedResources.get(index);
    }

    public ArrayList<Integer> getAllocated(){
        return allocatedResources;
    }

    public boolean isComplete(){
        return complete;
    }

    // other methods
    private ArrayList<Integer> stringToArray(String inputString){
        // attributes
        ArrayList<Integer> outputArray = new ArrayList<>();
        String s1 = inputString;

        // split
        String[] values = s1.split(",");

        // convert to an integer array list
        for (int i = 0; i < values.length; i++){
            outputArray.add(Integer.parseInt(values[i]));
        }

        return outputArray;
    }

    public void updateNeed(){
        int totalNeed = 0;
        // updates the needResources by subtracting allocatedResources from maxResources
        for (int i = 0; i < maxResources.size(); i++){
            neededResources.add(maxResources.get(i) - allocatedResources.get(i));
            // update total need
            totalNeed += neededResources.get(i);
        }
    }

    public void setComplete(Boolean complete){
        this.complete = complete;
    }




}
