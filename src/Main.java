import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Scanner;

/**
 * The Main class creates and runs the neural network built in the FirstTry class. It also inputs the information into the
 * constructor of the network and also features the weights matrix for the neural network. The class also has test cases to test
 * whether the neural network performs as expected.
 *
 * @author David Feng
 * @version February 20, 2020
 */

public class Main
{

   private static String networkInfoFile = "networkInfo.txt";
   private static String testCasesFile = "testCases.txt";

   /**
    * This method creates the FirstTry network, stores the weights matrix, and runs the four test cases for the neural network. It
    * also prints the outputs of the networks.
    *
    * @param args command line arguments
    */
   public static void main(String[] args) throws FileNotFoundException, IOException
   {
      if (args.length == 1)
      {
         networkInfoFile = args[0];
      }
      else if (args.length == 2)
      {
         networkInfoFile = args[0];
         testCasesFile = args[1];
      }

      System.out.println("Do you want to train the network [y/n]");
      Scanner readTrain = new Scanner(System.in);
      String trainOrNo = readTrain.nextLine();

      // read basic neural network values: number of input nodes, hidden layer array, and number of output nodes
      String fileName = networkInfoFile;
      Scanner sc = new Scanner(new FileReader(fileName));
      int inputNodes = sc.nextInt();
      int hiddenLayerLength = sc.nextInt();
      sc.nextLine();
      String[] hiddenLayerArray = sc.nextLine().split(",");
      int[] hiddenLayerInformation = new int[hiddenLayerLength];
      for (int i = 0; i < hiddenLayerLength; i++)
      {
         hiddenLayerInformation[i] = Integer.parseInt(hiddenLayerArray[i]);
      }
      int outputNodes = sc.nextInt();

      // read basic values necessary for training
      double minRandom = sc.nextDouble();
      double maxRandom = sc.nextDouble();
      double learningRate = sc.nextDouble();
      int epochs = sc.nextInt();
      double errorThreshold = sc.nextDouble();
      sc.nextLine();

      // set up weights matrix
      FirstTry network = new FirstTry(inputNodes, hiddenLayerInformation, outputNodes);

      // prints out basic information relevant to testing and training
      System.out.println();
      System.out.println("Basic Information");
      System.out.println("Number of layers: " + network.numLayers);
      System.out.println("Number of activations for each layer:");
      for (int activation = 0; activation < network.numLayers; activation++)
      {
         System.out.println("Layer " + (activation + 1) + ": " + network.layerInfo[activation] + " activations");
      }

      if (trainOrNo.equals("y"))
      {
         // prints out basic information
         System.out.println("Error threshold: " + errorThreshold);
         System.out.println("Max iterations allowed: " + epochs + " iterations");
         System.out.println("Learning factor: " + learningRate);
         System.out.println("Minimum weight: " + minRandom);
         System.out.println("Maximum weight: " + maxRandom);

         // Randomizing weights
         for (int n = 0; n < network.numLayers - 1; n++)
         {
            for (int j = 0; j < network.layerInfo[n]; j++)
            {
               for (int i = 0; i < network.layerInfo[n + 1]; i++)
               {
                  network.weights[n][j][i] = network.randomize(minRandom, maxRandom);
               }
            }
         }

         // read the test cases
         String testCasesData = testCasesFile;
         Scanner second = new Scanner(new FileReader(testCasesData));
         int numTestCases = second.nextInt();
         String[][] testCases = new String[numTestCases][inputNodes];
         second.nextLine();
         for (int testCase = 0; testCase < numTestCases; testCase++)
         {
            testCases[testCase] = second.nextLine().split(",");
         }

         // creates test cases array
         double[][] testCasesDouble = new double[numTestCases][inputNodes];
         for (int testCase = 0; testCase < numTestCases; testCase++)
         {
            for (int testCaseNode = 0; testCaseNode < inputNodes; testCaseNode++)
            {
               testCasesDouble[testCase][testCaseNode] = Double.parseDouble(testCases[testCase][testCaseNode]);
            }
         }

         // makes the expected values array
         String[][] expectedString = new String[numTestCases][outputNodes];
         double[][] expectedDouble = new double[numTestCases][outputNodes];
         for (int numCase = 0; numCase < numTestCases; numCase++)
         {
            expectedString[numCase] = second.nextLine().split(",");
            for (int node = 0; node < outputNodes; node++)
            {
               expectedDouble[numCase][node] = Double.parseDouble(expectedString[numCase][node]);
            }
         }

         // propagates the network
         System.out.println();
         System.out.println("Original training cases outputs: ");
         for (int propogateCase = 0; propogateCase < numTestCases; propogateCase++)
         {
            network.propagate(testCasesDouble[propogateCase]);

            //prints the output of the test cases
            for (int node = 0; node < outputNodes; node++)
            {
               System.out.println(network.activations[network.numLayers - 1][node]);
            }
         } // for (int propogateCase = 0; propogateCase < numTestCases; propogateCase++)

         // trains the network
         System.out.println();
         System.out.println("Post training outputs and expected outputs: ");
         network.train(numTestCases, testCasesDouble, expectedDouble, learningRate, epochs, errorThreshold);
         for (int propogateCase = 0; propogateCase < numTestCases; propogateCase++)
         {
            network.propagate(testCasesDouble[propogateCase]);

            System.out.println("Test Case: " + propogateCase);
            //prints the output of the test cases
            for (int node = 0; node < outputNodes; node++)
            {
               System.out.print("Post-training: " + network.activations[network.numLayers - 1][node]);
               System.out.print("  |  Expected: " + expectedDouble[propogateCase][node]);
               System.out.println();
            }
         } // for (int propogateCase = 0; propogateCase < numTestCases; propogateCase++)

         // prints out end weights
         network.printWeights();

         // prints learning rate
         System.out.println("Final learning rate: " + learningRate);

      } // if(trainOrNo.equals("y"))

      if (trainOrNo.equals("n"))
      {
         for (int i = 0; i < network.numLayers - 1; i++)
         {
            for (int j = 0; j < network.layerInfo[i]; j++)
            {
               String[] temp = sc.nextLine().split(",");
               double[] tempWeights = new double[network.layerInfo[i + 1]];
               for (int numberOfWeights = 0; numberOfWeights < network.layerInfo[i + 1]; numberOfWeights++)
               {
                  tempWeights[numberOfWeights] = Double.parseDouble(temp[numberOfWeights]);
               }
               network.weights[i][j] = tempWeights;
            }
         } // for (int i = 0; i < network.numLayers - 1; i++)
         // read the test cases
         String testCasesData = testCasesFile;
         Scanner second = new Scanner(new FileReader(testCasesData));
         int numTestCases = second.nextInt();
         String[][] testCases = new String[numTestCases][inputNodes];
         second.nextLine();
         for (int testCase = 0; testCase < numTestCases; testCase++)
         {
            testCases[testCase] = second.nextLine().split(",");
         }

         // creates test cases array
         double[][] testCasesDouble = new double[numTestCases][inputNodes];
         for (int testCase = 0; testCase < numTestCases; testCase++)
         {
            for (int testCaseNode = 0; testCaseNode < inputNodes; testCaseNode++)
            {
               testCasesDouble[testCase][testCaseNode] = Double.parseDouble(testCases[testCase][testCaseNode]);
            }
         }

         // makes the expected values array
         String[][] expectedString = new String[numTestCases][outputNodes];
         double[][] expectedDouble = new double[numTestCases][outputNodes];
         for (int numCase = 0; numCase < numTestCases; numCase++)
         {
            expectedString[numCase] = second.nextLine().split(",");
            for (int node = 0; node < outputNodes; node++)
            {
               expectedDouble[numCase][node] = Double.parseDouble(expectedString[numCase][node]);
            }
         }

         // propagates the network
         System.out.println();
         System.out.println("Original test cases outputs: ");
         for (int propogateCase = 0; propogateCase < numTestCases; propogateCase++)
         {
            network.propagate(testCasesDouble[propogateCase]);

            //prints the output of the test cases
            for (int node = 0; node < outputNodes; node++)
            {
               System.out.print("Original: " + network.activations[network.numLayers - 1][node]);
               System.out.print("  |  Expected: " + expectedDouble[propogateCase][node]);
               System.out.println();
            }
         } // for (int propogateCase = 0; propogateCase < numTestCases; propogateCase++)

         // prints out weights
         System.out.println();
         System.out.println("Weights:");
         network.printWeights();
      } // if(trainOrNo.equals("n"))
   } // public static void main(String[] args) throws FileNotFoundException, IOException
} // public class Main
