import java.io.FileReader;
import java.io.IOException;
import java.util.Scanner;

public class DianeMain
{

   private static String networkInfoFileDD = "networkInfo.txt";
   private static String testCasesFileDD = "testCases.txt";

   /**
    * This method creates the FirstTry network, stores the weights matrix, and runs the four test cases for the neural network. It
    * also prints the outputs of the networks.
    *
    * @param args command line arguments
    */
   public static void main(String[] args) throws IOException
   {
      if (args.length == 1)
      {
         networkInfoFileDD = args[0];
      }
      else if (args.length == 2)
      {
         networkInfoFileDD = args[0];
         testCasesFileDD = args[1];
      }

      DibDump dibdumper = new DibDump();
      int[][] activationBMP = dibdumper.BMPtoArray("test1.bmp");
      int length = activationBMP.length * activationBMP[0].length;
      double[] activationInput = new double[length];
      int count = 0;
      for (int i = 0; i < activationBMP.length; i++)
      {
         for (int j = 0; j < activationBMP[i].length; j++)
         {
            activationBMP[i][j] = dibdumper.colorToGrayscale(activationBMP[i][j]);
            activationBMP[i][j] &= 0x00ffffff;
            activationInput[count] = activationBMP[i][j];
            double scalingfactor = 1 << 24;
            activationInput[count] /= scalingfactor;
            count++;
         }
      }

      System.out.println("Do you want to train the network [y/n]");
      Scanner readTrain = new Scanner(System.in);
      String trainOrNo = readTrain.nextLine();

      // read basic neural network values: number of input nodes, hidden layer array, and number of output nodes
      String fileName = networkInfoFileDD;
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

      FirstTry network = new FirstTry(length, hiddenLayerInformation, length);

      // read basic values necessary for training
      double minRandom = sc.nextDouble();
      double maxRandom = sc.nextDouble();
      double learningRate = sc.nextDouble();
      int epochs = sc.nextInt();
      double errorThreshold = sc.nextDouble();
      int pixels = sc.nextInt();
      sc.nextLine();

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


         double[][] testCasesDouble = new double[1][length];
         testCasesDouble[0] = activationInput;
         network.train(1, testCasesDouble, testCasesDouble,learningRate, epochs, errorThreshold);

         for (int propagateCase = 0; propagateCase < 1; propagateCase++)
         {
            network.propagate(testCasesDouble[propagateCase]);

            for (int node = 0; node < outputNodes; node++)
            {
               System.out.print("Original: " + network.activations[network.numLayers - 1][node]);
               System.out.print("  |  Expected: " + testCasesDouble[propagateCase][node]);
               System.out.println();
            }
         } // for (int propogateCase = 0; propogateCase < numTestCases; propogateCase++)

         double[] oneDimArray = network.activations[3];
         int[][] finalArray = new int[pixels][pixels];
         int counter = 0;
         for (int i = 0; i < pixels; i++)
         {
            for (int j = 0; j < pixels; j++)
            {
               oneDimArray[counter] = oneDimArray[counter] * (1 << 24);
               finalArray[i][j] = (int) oneDimArray[counter];
               counter++;
            }
         }

         dibdumper.arrayToBMP(finalArray, "whatever.bmp");

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

         // propagates the network
         System.out.println();
         System.out.println("Original test cases outputs: ");

         double[][] testCasesDouble = new double[1][length];
         testCasesDouble[0] = activationInput;

         for (int propagateCase = 0; propagateCase < 1; propagateCase++)
         {
            network.propagate(testCasesDouble[propagateCase]);

            //prints the output of the test cases
            for (int node = 0; node < outputNodes; node++)
            {
               System.out.print("Original: " + network.activations[network.numLayers - 1][node]);
               System.out.print("  |  Expected: " + testCasesDouble[propagateCase][node]);
               System.out.println();
            }
         } // for (int propogateCase = 0; propogateCase < numTestCases; propogateCase++)

         double[] oneDimArray = network.activations[3];
         int[][] finalArray = new int[101][101];
         int counter = 0;
         for (int i = 0; i < 101; i++)
         {
            for (int j = 0; j < 101; j++)
            {
               oneDimArray[counter] = oneDimArray[counter] * (1 << 24);
               finalArray[i][j] = (int) oneDimArray[counter];
               finalArray[i][j] = dibdumper.colorToGrayscale(finalArray[i][j]);
               counter++;
            }
         }

         dibdumper.arrayToBMP(finalArray, "whatever.bmp");

         // prints out weights
         System.out.println();
         System.out.println("Weights:");
         network.printWeights();
      }
   }
}
