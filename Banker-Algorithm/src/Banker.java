import java.io.*;
import java.util.ArrayList;
import java.util.Random;

public class Banker {
    //attributes
    private String fileName;
    private ArrayList<Process> originalProcesses = new ArrayList<>();
    private ArrayList<Process> processes = new ArrayList<>();
    private ArrayList<Integer> availableResources = new ArrayList<>();
    private ArrayList<Integer> systemResources = new ArrayList<>();
    private ArrayList<String> pathsToCompleteion = new ArrayList<>();
    int totalProcesses;
    int totalResources;

    //constructor
    Banker(String fileName){
        // take a file name string and creates a Banker from that file
        this.fileName = fileName + ".txt";

        // create a file
        String currentDirectory = System.getProperty("user.dir");
        File file = new File( currentDirectory + "\\src\\" + this.fileName);

        // create file reader
        FileReader in;

        // load file into scanner
        try{
            in = new FileReader(file);
        } catch (FileNotFoundException ex){
            System.out.println("File not found.  Running default file.");

            file = new File( currentDirectory + "\\src\\input.txt");
            try{
                in = new FileReader(file);
            } catch (FileNotFoundException e){
                System.out.println("File not found.");
                return;
            }
        }

        // read the file
        String line;
        int index = -1;
        BufferedReader br = new BufferedReader(in);

        // read each line and create processes
        try {
            while((line = br.readLine()) != null){
                if (index == -1){
                    // set the total resources for the system
                    setSystemResources(line);
                } else {
                    // create the processes
                    String[] processData = line.split(":");
                    processes.add(new Process(index, processData[0], processData[1]));
                }
                index += 1;
            }

            // set a few attributes
            setAvailableResources();
            totalResources = availableResources.size();
            totalProcesses = processes.size();

        } catch (IOException ex){

            System.out.println(ex.toString());
        }
    }

    /*
    Methods
     */

    private void setSystemResources(String input){
        // builds a system resource array from a string
        String[] resources = input.split(",");

        // convert to an integer array list
        for (int i = 0; i < resources.length; i++){
            systemResources.add(Integer.parseInt(resources[i]));
        }
    }

    private void setAvailableResources(){
        // sets the available resources
        availableResources = new ArrayList<>();
        for (int i = 0; i < systemResources.size(); i++){
            int sum = 0;

            // calculates the total available resources
            for (int j = 0; j < processes.size(); j++){
                sum += processes.get(j).getAllocated(i);
            }
            availableResources.add(systemResources.get(i) - sum);
        }
    }

    public String getPathToCompletion(){
        // takes a start index as a start process and calculates a path to completion
        ArrayList<Integer> processIndexes = new ArrayList<>();
        String processCompletionOrder = new String();
        int index = 0;

        // iterate over processes and check with resources to see if they can be completed
        while (index < processes.size()){
            for(int i = 0; i < processes.size(); i++){
                if (processes.get(i).isComplete() == false){
                    if(compareNeeds(processes.get(i).getNeed()) == true){
                        index += 1;
                        // add the index to processIndexes
                        processIndexes.add(i);

                        // add resources to process and updated process needs
                        processes.get(i).setComplete(true);

                        // update available resources
                        updateAvailableResources(processes.get(i).getAllocated());
                    }
                }
            }
        }

        // turn processIndexes into a string
        StringBuilder sb = new StringBuilder();
        for(int i = 0; i < processIndexes.size(); i++){
            sb.append("P" + processes.get(processIndexes.get(i)).getId() + " ");
        }

        processCompletionOrder = sb.toString();

        return processCompletionOrder;
    }

    public boolean compareNeeds(ArrayList<Integer> needs){
        // takes a process and compares its needs to available resources
        boolean adequateResources = true;

        for (int i = 0; i < availableResources.size();i++){
            if(availableResources.get(i) < needs.get(i)){
                adequateResources = false;
                break;
            }
        }
        // if available resources > needs return true
        return adequateResources;
    }

    public void updateAvailableResources(ArrayList<Integer> assigned){
        // takes an array list of assigned resources from a process
        for (int i = 0; i < availableResources.size(); i++){
            availableResources.set(i, (availableResources.get(i) + assigned.get(i)));
        }
    }

    private void shuffleProcesses() {
        // get size of the list
        int totalElements = processes.size();
        // initialize random number generator
        Random random = new Random();
        for (int i = 0; i < totalElements; i++) {
            // get the list element at current index
            Process currentElement = processes.get(i);
            // generate a random index within the range of list size
            int randomIndex = i + random.nextInt(totalElements - i);
            // set the element at current index with the element at random
            // generated index
            processes.set(i, processes.get(randomIndex));
            // set the element at random index with the element at current loop
            // index
            processes.set(randomIndex, currentElement);
        }
    }

    public void getAllPaths(){
        // attributes
        long start = System.currentTimeMillis();
        long end = start + 60*1000;

        // get the first path to completion
        String validPath = getPathToCompletion();
        pathsToCompleteion.add(validPath);
        System.out.println(validPath);

        while (System.currentTimeMillis() < end) {
            // restart
            resetProcesses();
            setAvailableResources();

            // shuffle the processes
            shuffleProcesses();

            // check to make sure the validPath to completion
            validPath = getPathToCompletion();
            if (checkPaths(validPath) == false){
                pathsToCompleteion.add(validPath);
                System.out.println(validPath);

            }


        }
    }

    private void resetProcesses(){
        for (int i = 0; i < processes.size(); i++) {
            processes.get(i).setComplete(false);
        }
    }

    private boolean checkPaths(String path){
        // checks all paths in an array and returns true if there is a matching path
        boolean match = false;
        for (int i = 0; i < pathsToCompleteion.size(); i++){
            if(pathsToCompleteion.get(i).matches(path)){
                match = true;
            }
        }
        return match;
    }
}
