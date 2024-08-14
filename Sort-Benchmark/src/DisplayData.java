/**
 File Name: DisplayData.java
 Author(s): Brian Richardson
 Date: 7/13/2021
 Description:  chooses a text file and creates a JTable
 */
import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;


@SuppressWarnings("serial")
public class DisplayData extends JFrame {
    private JTable table;
    private String[][] data;
    JPanel display;

    private JButton button;

    DisplayData() {
        super();
        initialize();
    }

    private void initialize() {
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // set exit condition
        setSize(500, 500);
        setTitle("Results");
        final Dimension screen = Toolkit.getDefaultToolkit().getScreenSize(); // get user screen size
        setLayout(new FlowLayout());
        display = new JPanel(new GridLayout(1, 1));
        display.setSize(800, 800);
        setContentPane(display);

        // add components
        final JMenuBar menuBar = new JMenuBar();
        fileMenu(menuBar);
        setJMenuBar(menuBar);

        setLocation((screen.width - getWidth()) / 2, (screen.height - getHeight()) / 2); // FIXED IT screen.width used twice
        setVisible(true);
    }

    private void fileMenu(final JMenuBar menuBar) {
        final JMenu fileMenu = new JMenu("File");
        final JMenuItem open = new JMenuItem("Open");
        fileMenu.add(open);
        menuBar.add(fileMenu);

        open.addActionListener(new ActionListener() {

            public void actionPerformed(ActionEvent ignored) {
                final JFileChooser fileChooser = new JFileChooser();
                fileChooser.setCurrentDirectory(new java.io.File("."));
                FileNameExtensionFilter filter = new FileNameExtensionFilter("TEXT FILES", "txt", "text");
                fileChooser.setFileFilter(filter);
                // show open dialogue
                if (fileChooser.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
                    // use the file
                    final File selectedFile = fileChooser.getSelectedFile();
                    System.out.println(selectedFile.getAbsolutePath());
                    try {
                        readFile(selectedFile);
                    } catch (IOException ioException) {
                        ioException.printStackTrace();
                    }
                }
            }
        });
    }

    private void readFile(final File file) throws IOException {
        ArrayList<String> lines = new ArrayList<>();

        try {

            final FileReader in = new FileReader(file);
            String line;
            final BufferedReader bufferedReader = new BufferedReader(in);
            // append each line to the text area
            while ((line = bufferedReader.readLine()) != null) {
                lines.add(line);
            }
            bufferedReader.close();
        } catch (FileNotFoundException ignored) {
            JOptionPane.showMessageDialog(this, "File not found");
        }

        // create a new data set
        data = new String[lines.size()][5];

        for(int i = 0; i < lines.size(); i++){
            String[] splitLine = lines.get(i).split(",");

            for (int j = 0; j < splitLine.length; j++){
                data[i][j] = splitLine[j];
            }
        }

        String[] columnNames = {"Size", "Avg Count", "Count Coef", "Avg Time", "Time Coef"};
        table = new JTable(data, columnNames);

        JScrollPane tableContainer = new JScrollPane(table);

        display.add(tableContainer, BorderLayout.CENTER);
        display.updateUI();

    }

    public static void main(String[] ignored) {
        new DisplayData();
    }
}
