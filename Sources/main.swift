//
//  NeuralNet-MNIST.swift
//  Swift-AI
//
//  Created by Collin Hundley on 4/8/17.
//
//

import Foundation
import NeuralNet


/*

    INTRODUCTION
 
        This package contains a NeuralNet training example for the MNIST handwriting dataset.
    
    USAGE
 
        1) Replace the paths below with the appropriate location on your system.
           mnistDataDir: must point to the directory containing the MNIST data (included in this repo).
           outputFilepath: will be the name and location of the trained network file. You may save it anywhere.
 
        2) Run the application.
           Make sure you're running in Release mode with Whole Module Optimization enabled.
           Training will take < 1 minute on most modern hardware.
 
        3) When training completes, it will save a copy of the trained network to disk at your chosen location.
           You may use this file to recreate the network in other applications.
 
    NOTES
    
        The network below has been pre-populated with reasonable parameters for the MNIST set.
        Try tuning the parameters yourself! You can try different minibatch sizes, network dimensions,
        activation functions, learning rates, momentum factors, etc.
 
        For fully-connected neural nets, current state-of-the-art performance on MNIST is approximately 98.4% (1.6% error).
        With proper tuning and a little patience, you can achieve state-of-the-art performance with this package.
        Note: the jump from 98% to 98.4% is fairly significant, so the training routine exits at 98% by default.
        This allows most machines to complete training in under 60 seconds.

*/


// ADD CUSTOM PATHS HERE ---------------------------------------------------

/// Path to MNIST dataset directory.
let mnistDataDir = "/PATH/TO/NeuralNet-MNIST/MNIST"

/// Full filepath for trained network output file.
let outputFilepath = "/PATH/TO/neuralnet-mnist-trained"

// -------------------------------------------------------------------------


do {
    
    // Extract MNIST data from disk
    
    print("Extracting MNIST data...")
    let manager = try MNISTManager(directory: mnistDataDir,
                                   pixelRange: (min: 0, max: 1), // White pixels 0, black pixels 1
                                   batchSize: 100)
    
    // Create dataset for training
    
    let dataset = try NeuralNet.Dataset(trainInputs: manager.trainImages, trainLabels: manager.trainLabels,
                                        validationInputs: manager.validationImages, validationLabels: manager.validationLabels)

    // Define network structure and create neural net
    
    print("Creating neural network...")
    let structure = try NeuralNet.Structure(nodes: [784, 500, 10],
                                            hiddenActivation: .rectifiedLinear, outputActivation: .softmax,
                                            batchSize: 100, learningRate: 0.8, momentum: 0.9)
    let nn = try NeuralNet(structure: structure)
    
    // Begin training routine
    
    print("\n----------------- BEGINNING MNIST TRAINING -----------------")
    let (epochs, error) = try nn.train(dataset, maxEpochs: 50, errorThreshold: 0.02, errorFunction: .percentage) { (epoch, err) -> Bool in
        
        // Log progress
        let percCorrect = (1 - err) * 100
        let percError = err * 100
        print("\nEpoch \(epoch)")
        print("Accuracy:\t\(percCorrect)%")
        print("Error:\t\t\(percError)%")
        
        // Decay learning rate and momentum
        nn.learningRate *= 0.97
        nn.momentumFactor *= 0.97
        
        // Allow training to continue
        return true
    }
    
    // Save net to disk and log result
    try nn.save(to: URL(fileURLWithPath: outputFilepath))
    print("\n--------------------------- DONE ---------------------------")
    print("\nFinal accuracy: \((1 - error) * 100)%")
    print("Trained network stored at: \(outputFilepath)\n\n")
    
} catch {
    print(error)
}

