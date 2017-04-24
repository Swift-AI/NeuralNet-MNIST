// swift-tools-version:3.1

import PackageDescription

let package = Package(
    name: "NeuralNet-MNIST",
    dependencies: [
        .Package(url: "https://github.com/Swift-AI/NeuralNet.git", majorVersion: 0, minor: 3)
    ]
)
