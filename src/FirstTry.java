import java.io.*;

/**
 * Neural Network
 *
 * This class constructs one flexible neural network with a variable amount of input nodes, hidden activation layers, number of
 * nodes per hidden activation layer, and output nodes. The connectivity of the neural network is a fully-connected and
 * feed-forward such that every node in one layer is connected to every node in the next layer.
 *
 * This neural network can be constructed by inputting the number of input nodes, an array description of hidden layers, and the
 * number of output nodes. This information will be used to construct the 2D matrix of activations and 3D matrix of weights for
 * the neural network. The network is initiated by a propagate function, which calculates the output of the network based on the
 * inputs.
 *
 * This network can be trained with three activation layers or fewer; specifically, the network uses gradient descent and implementing
 * backpropagation to minimize the average of all the individual error functions, changing the weights while it performs that task.
 *
 * Methods in this class
 * void   propagate
 * double outputFunction
 * double outputFunctionPrime
 * void   train
 * int    checkTrain
 * double randomize
 * void   printWeights
 *
 * @author David Feng
 * @version March 02, 2020
 */
public class FirstTry
{

   int inputs;
   int numLayers;
   int[] layerInfo;
   int[] hidden;
   int output;
   double[][] activations;
   double[][][] weights;
   double[][] thetaArray;
   double[][] psiArray;
   double[][] omegaArray;

   /**
    * Constructs a FirstTry object that is the feed-forward, fully-connected neural network. The neural network takes in an integer
    * number of input nodes, an integer array of hidden layer nodes to indicate the number of nodes in every hidden layer, and an
    * integer number of output nodes. The constructor initializes the global variables and generates the activations and weights
    * matrix of the neural network as well.
    *
    * @param inputNodes       the number of input nodes in the neural network
    * @param hiddenLayerNodes an array of integers that indicates the number hidden layers and the number of nodes in every hidden
    *                         layer
    * @param outputNodes      the number of output nodes in the neural network
    */
   public FirstTry(int inputNodes, int[] hiddenLayerNodes, int outputNodes)
   {
      //creates the global variables
      inputs = inputNodes;
      hidden = hiddenLayerNodes;
      output = outputNodes;
      numLayers = hiddenLayerNodes.length + 2;

      //creates the layer info array
      layerInfo = new int[numLayers];
      layerInfo[0] = inputs;
      for (int layer = 1; layer < numLayers - 1; layer++)
      {
         layerInfo[layer] = hiddenLayerNodes[layer - 1];
      }
      layerInfo[numLayers - 1] = outputNodes;


      //calculates the activations matrix for the network
      activations = new double[hiddenLayerNodes.length + 2][];
      activations[0] = new double[inputNodes];
      activations[hiddenLayerNodes.length + 1] = new double[outputNodes];
      for (int hiddenLayer = 1; hiddenLayer < hiddenLayerNodes.length + 1; hiddenLayer++)
      {
         activations[hiddenLayer] = new double[hiddenLayerNodes[hiddenLayer - 1]];
      }

      //creates the weights matrix for the neural network
      weights = new double[hiddenLayerNodes.length + 1][][];
      weights[0] = new double[inputNodes][hiddenLayerNodes[0]];
      weights[hiddenLayerNodes.length] = new double[hiddenLayerNodes[hiddenLayerNodes.length - 1]][outputNodes];
      for (int hiddenNode = 1; hiddenNode < hiddenLayerNodes.length; hiddenNode++)
      {
         weights[hiddenNode] = new double[hiddenLayerNodes[hiddenNode - 1]][hiddenLayerNodes[hiddenNode]];
      }

      //creates the backpropagation arrays
      thetaArray = new double[numLayers][];
      omegaArray = new double[numLayers][];
      psiArray = new double[numLayers][];
      for (int layer = 0; layer < numLayers; layer++)
      {
         thetaArray[layer] = new double[layerInfo[layer]];
         omegaArray[layer] = new double[layerInfo[layer]];
         psiArray[layer] = new double[layerInfo[layer]];
      }
   } // public FirstTry(int inputNodes, int[] hiddenLayerNodes, int outputNodes)

   /**
    * The propagate method helps generate the activations in the hidden layers of the neural network and calculates the final
    * output of the system. The method loops over the layers of the neural network.
    *
    * @param inputs a double array of the inputs to the neural network
    */
   public void propagate(double[] inputs)
   {
      activations[0] = inputs;
      for (int n = 1; n < numLayers; n++)
      {
         for (int k = 0; k < layerInfo[n]; k++)
         {
            double temp = 0.0;

            for (int m = 0; m < layerInfo[n - 1]; m++)
            {
               temp += weights[n - 1][m][k] * activations[n - 1][m];
            }
            thetaArray[n][k] = temp;

            activations[n][k] = outputFunction(temp);
         }
      } // for (int n = 1; n < numLayers; n++)
   } // public void propagate(double[] inputs)

   /**
    * This is the output function applied to each node to produce the final output
    *
    * @param x the input that is passed into the output function
    * @return the result of the output function
    */
   public double outputFunction(double x)
   {
      return 1.0 / (1.0 + Math.exp(-x));
   }

   /**
    * The derivative of the output function, which is integral in the training process
    *
    * @param x the input that is passed into this derivative function
    * @return the result of the output derivative function
    */
   public double outputFunctionPrime(double x)
   {
      double output = outputFunction(x);
      return output * (1.0 - output);
   }

   /**
    * This method trains the neural network to optimize the weights for a given set of test cases using gradient descent, where
    * one goes "down hill" in weight space. The goal of this function is to minimize the error function in the network.
    *
    * @param numTrainCases the number of training cases being used in the method
    * @param inputs        the training cases used in optimizing the weights
    * @param expected      the expected results for the the training cases
    * @param learningRate  an predetermined rate at which the network learns
    * @param epochs        the number of iterations that the training function uses
    */
   public void train(int numTrainCases, double[][] inputs, double[][] expected, double learningRate, int epochs, double errorThreshold)
   {
      for (int testCase = 0; testCase < numTrainCases; testCase++)
      {
         propagate(inputs[testCase]);
         calculateBackprop(expected, testCase, learningRate);
      }

      double error = 0.0;

      for (int cases = 0; cases < numTrainCases; cases++)
      {
         for (int outputs = 0; outputs < output; outputs++)
         {
            error += 0.5 * ((expected[cases][outputs] - activations[numLayers - 1][outputs]) * (expected[cases][outputs] - activations
                    [numLayers - 1][outputs]));
         }
      }

      int numLoops = 1;
      error = error / (numTrainCases * output);

      while (checkTrain(numLoops, epochs, error, errorThreshold) == -2)
      {
         error = 0.0;

         for (int testCase = 0; testCase < numTrainCases; testCase++)
         {
            propagate(inputs[testCase]);
            calculateBackprop(expected, testCase, learningRate);

            for (int outputs = 0; outputs < output; outputs++)
            {
               error += 0.5 * ((expected[testCase][outputs] - activations[numLayers - 1][outputs]) * (expected[testCase][outputs] - activations
                       [numLayers - 1][outputs]));
            }
         } // for (int testCase = 0; testCase < numTrainCases; testCase++)
         error = error / (numTrainCases * output);
         numLoops++;
      } // while (checkTrain(numLoops, epochs, error, errorThreshold) == -2)
   } //public void train(int numTrainCases, double[][] inputs, double[][] expected, double learningRate, int epochs, double errorThreshold)

   /**
    * Calculates the omega, psi, and weights in the backpropagation algorithm
    *
    * @param expected     the expected values from training
    * @param testCase     the test case number, which is useful for finding the expected value
    * @param learningRate the learning rate for the network
    */
   public void calculateBackprop(double[][] expected, int testCase, double learningRate)
   {
      for (int outputNode = 0; outputNode < output; outputNode++)
      {
         omegaArray[numLayers - 1][outputNode] = expected[testCase][outputNode] - activations[numLayers - 1][outputNode];
         psiArray[numLayers - 1][outputNode] = omegaArray[numLayers - 1][outputNode] * outputFunctionPrime(thetaArray[numLayers - 1][outputNode]);

         for (int j = 0; j < layerInfo[numLayers - 2]; j++)
         {
            weights[numLayers - 2][j][outputNode] += learningRate * activations[numLayers - 2][j] * psiArray[numLayers - 1][outputNode];
         }
      } // for (int outputNode = 0; outputNode < output; outputNode++)

      for (int layerNum = numLayers - 2; layerNum > 0; layerNum--)
      {
         for (int j = 0; j < layerInfo[layerNum]; j++)
         {
            double omegaSum = 0.0;

            for (int i = 0; i < layerInfo[layerNum + 1]; i++)
            {
               omegaSum += psiArray[layerNum + 1][i] * weights[layerNum][j][i];
            }
            psiArray[layerNum][j] = omegaSum * outputFunctionPrime(thetaArray[layerNum][j]);

            for (int k = 0; k < layerInfo[layerNum - 1]; k++)
            {
               weights[layerNum - 1][k][j] += learningRate * activations[layerNum - 1][k] * psiArray[layerNum][j];
            }
         }
      } // for (int layerNum = numLayers - 2; layerNum > 0; layerNum--)
   } // public void calculateBackprop(double[][] expected, int testCase, double learningRate)

   /**
    * Checks whether the training for the network is done given the number of intended epochs and the average error
    *
    * @param epochs    the number of current iterations
    * @param goalEpoch the number of maximum iterations under which the network will stop training
    * @param error     the error of the network under the current set of weights
    * @param goalError the error threshold of the network
    * @return -2 if neither condition (number of epochs, average error) is met
    * -1 if the error threshold is reached
    * 0 if the number of epochs is reached
    */
   public int checkTrain(int epochs, int goalEpoch, double error, double goalError)
   {
      int returnable = -2;

      if (epochs >= goalEpoch)
      {
         System.out.println("Training complete because goal epochs of " + goalEpoch + " was reached with error of " + error);
         returnable = 0;
      }

      if (error <= goalError)
      {
         System.out.println("Training complete because network reached error threshold with " + epochs + " training cases with " +
                 "error of " + error);
         returnable = -1;
      }

      return returnable;
   } // public int checkTrain(int epochs, int goalEpoch, double error, double goalError)

   /**
    * Generates a random double between two user-determined values using the Math.random function
    *
    * @param min the minimum value of the random number
    * @param max the maximum value of the random number
    * @return the random number
    */
   public double randomize(double min, double max)
   {
      return (max - min) * Math.random() + min;
   }

   /**
    * Prints the current weights in the neural network into a file and labels them using the conventional method for identifying weights
    */
   public void printWeights() throws IOException
   {
      PrintWriter printer = new PrintWriter(new BufferedWriter(new FileWriter("weights.txt")));
      for (int n = 0; n < numLayers - 1; n++)
      {
         for (int j = 0; j < layerInfo[n]; j++)
         {
            for (int i = 0; i < layerInfo[n + 1]; i++)
            {
               printer.println("w" + n + j + i + " = " + weights[n][j][i]);
            }
         }
      }
      printer.close();
   } // public void printWeights() throws IOException

} // public class FirstTry