import java.util.Scanner;

public class Main {
    public static void main(String[] args){
        Scanner userInput = new Scanner(System.in); // create scanner object
        System.out.println("Please enter the file name: ");
        String fileName = userInput.nextLine();  // get user input

        Banker banker = new Banker(fileName);
        //System.out.println(banker.getPathToCompletion());
        banker.getAllPaths();
    }
}
