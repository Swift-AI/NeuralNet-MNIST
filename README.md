![Banner](https://github.com/Swift-AI/Swift-AI/blob/master/SiteAssets/logo/banner.png)

# NeuralNet-MNIST

An MNIST handwriting training example for the [NeuralNet](https://github.com/Swift-AI/NeuralNet) package.

This application is part of the Swift AI project. Full details on the project can be found in the [main repo](https://github.com/Swift-AI/Swift-AI).

## Installation

1. Clone the repository:

```sh
git clone https://github.com/Swift-AI/NeuralNet-MNIST.git
```

2. Generate Xcode project:

```sh
swift package generate-xcodeproj
```

## Training

### Setup

This project comes packaged with training data and a pre-built routine. The only thing you need to do is edit the following lines at the top of `main.swift`:

```swift
// Path to MNIST dataset directory.
let mnistDataDir = "/PATH/TO/NeuralNet-MNIST/MNIST"

// Full filepath for trained network output file.
let outputFilepath = "/PATH/TO/neuralnet-mnist-trained"
```

You should set `mnistDataDir` to the absolute path of the training data directory (included in this repository). This is necessary because Swift Package Manager currently doesn't support app bundles.

`outputFilepath` will be the location that the final, trained network is stored. You may set it to whatever you like.

### Run

Once these paths are set, just hit run and watch!

***Always run the trainer in release mode!*** or it will be a long day :)

### Customization

You can customize a number of parameters in the trainer and and neural network. The [NeuralNet](https://github.com/Swift-AI/NeuralNet) package contains more information on how to construct a neural net, so we won't go into detail here.

See the information at the top of `main.swift` for more inspiration.
 
## Data

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) is used for training. This includes 70,000 handwriting samples of the digits 0-9.





