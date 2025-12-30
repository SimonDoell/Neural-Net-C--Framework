#include <list>
#include <string>
#include <algorithm>
#include <vector>
#include <iostream>
#include <cmath>
#include <random>
#include <cstdlib>
// #include <omp.h>







// *Functions
float getRandomWeight() {
    float detail = 10000.0f;
    float randNum = (((rand() % int(detail)) / detail) - 0.5f);
    return randNum;
}

float getStartBias() {return 0.0f;}

float Sigmoid(float x) {return 1.0f / (1.0f + exp(-x));}
float SigmoidDerivative(float x) {
    float s = Sigmoid(x);
    return s * (1 - s);
}


float ReLU(float x) {return x < 0 ? 0 : x;}
float ReLUDerivative(float x) {return x < 0 ? 0 : 1;}

float leakyReLU(float x) {return std::max(x*0.1f, x);}
float leakyReLUDerivative(float x) {return x <= 0 ? 0.1f : 1.0f;}

float leakySigmoidAlpha = 0.05f;
float leakySigmoid(float x) {return Sigmoid(x) + leakySigmoidAlpha * x;}
float leakySigmoidDerivative(float x) {
    float s = Sigmoid(x);
    return s * (1.0f - s) + leakySigmoidAlpha;
}

float Tanh(float x) {
    float e1 = exp(x);
    float e2 = exp(-x);
    return (e1 - e2) / (e1 + e2);
}
float TanhDerivative(float x) {
    float t = Tanh(x);
    return 1.0 - (t * t);
}




float calcActivationFunctionDerivative(float(*_activationFunction)(float), float x) {
    if (_activationFunction == ReLU) return ReLUDerivative(x);
    else if (_activationFunction == Sigmoid) return SigmoidDerivative(x);
    else if (_activationFunction == leakyReLU) return leakyReLUDerivative(x);
    else if (_activationFunction == leakySigmoid) return leakySigmoidDerivative(x);
    else if (_activationFunction == Tanh) return TanhDerivative(x);
    else return x;
}




struct Neuron {
    public:
        float activation;
        float bias;
        std::vector<float> weights;

        float preActivation;
        float delta;

        Neuron(int _nextLayerNeuronAmount) : bias(getStartBias()) {for (int i = 0; i < _nextLayerNeuronAmount; ++i) weights.emplace_back(getRandomWeight());}
};



struct Layer {
    public:
        int amountNeurons;
        std::vector<Neuron> neurons;
        std::vector<Neuron*> nextLayerNeurons;
        float(*activationFunc)(float);

        Layer() {}
        Layer(int _amountNeuron, std::vector<Neuron*> _nextLayerNeurons, float(*_activationFunc)(float) = ReLU): nextLayerNeurons(_nextLayerNeurons), amountNeurons(_amountNeuron), activationFunc(_activationFunc) {
            for (int i = 0; i < _amountNeuron; ++i) neurons.emplace_back(Neuron(_nextLayerNeurons.size()));
        }

        std::vector<Neuron*> getNeuronsAsPtr() {
            std::vector<Neuron*> pointers;
            for (int i = 0; i < neurons.size(); ++i) pointers.emplace_back(&neurons[i]);
            return pointers;
        }

        void calculateNextLayer(float(*nextLayerActivationFunc)(float)) {
            // Weights
            for (int x = 0; x < nextLayerNeurons.size(); ++x) {
                nextLayerNeurons[x]->activation = 0;

                for (int i = 0; i < neurons.size(); ++i) {
                    nextLayerNeurons[x]->activation += neurons[i].activation * neurons[i].weights[x];
                }
            }

            for (int i = 0; i < nextLayerNeurons.size(); ++i) {
                // Bias
                nextLayerNeurons[i]->activation += nextLayerNeurons[i]->bias;

                // Activation Function
                nextLayerNeurons[i]->preActivation = nextLayerNeurons[i]->activation;
                nextLayerNeurons[i]->activation = nextLayerActivationFunc(nextLayerNeurons[i]->activation);
            }
        }
};



struct ANN {
    private:
        void calculateDelta(const std::vector<float>& desiredValuesForOutputlayerIndex) {
            if (layers.back().neurons.size() != desiredValuesForOutputlayerIndex.size()) std::cout << "\n\n-----List size mismatch in backprop!-----\n\n";

            // Calculate delta
            for (int i = layers.size()-1; i >= 0; --i) {
                for (int x = 0; x < layers[i].neurons.size(); ++x) {
                    Neuron& currNeuron = layers[i].neurons[x];

                    if (i == layers.size()-1) {
                        currNeuron.delta = (currNeuron.activation - desiredValuesForOutputlayerIndex[x]) * calcActivationFunctionDerivative(layers[i].activationFunc, currNeuron.preActivation);
                        // Delta = error * funcDerivative
                    } else {
                        float sum = 0;
                        for (int w = 0; w < layers[i].nextLayerNeurons.size(); ++w) 
                            sum += currNeuron.weights[w] * layers[i].nextLayerNeurons[w]->delta;
                        
                        currNeuron.delta = sum * calcActivationFunctionDerivative(layers[i].activationFunc, currNeuron.preActivation);
                        // --> Sum is all the deltas of the next Layer and how string the current Neuron is responsible for them according to the weights
                    }
                }
            }
        }

        void calculateBackpropChanges() {
            // Backpropagation
            for (int l = 0; l < layers.size(); ++l) {
                for (int n = 0; n < layers[l].neurons.size(); ++n) {
                    for (int w = 0; w < layers[l].neurons[n].weights.size(); ++w) {
                        layers[l].neurons[n].weights[w] -= learningRate * layers[l].neurons[n].activation * layers[l].nextLayerNeurons[w]->delta;
                    }
                }
            }

            for (int l = 0; l < layers.size(); ++l) {
                for (int n = 0; n < layers[l].neurons.size(); ++n) {
                    layers[l].neurons[n].bias -= learningRate * layers[l].neurons[n].delta;
                }
            }
        }

    public:
        std::vector<Layer> layers;
        std::vector<int> layersNeuronAmount;
        float learningRate = 0.01f;

        ANN(std::vector<int> _layersNeuronAmount, float(*_activationFunctionOutputLayer)(float) = Sigmoid, float(*_activationFunctionHiddenLayers)(float) = ReLU) : layersNeuronAmount(_layersNeuronAmount) {
            layers.resize(_layersNeuronAmount.size());
            layers.back() = (Layer(_layersNeuronAmount.back(), {}, _activationFunctionOutputLayer));  // --> Output Layer, no next Neurons

            for (int i = _layersNeuronAmount.size()-2; i >= 0; --i) 
                layers[i] = Layer(_layersNeuronAmount[i], layers[i+1].getNeuronsAsPtr(), _activationFunctionHiddenLayers);
        }

        std::vector<float> getOutputValues() {
            std::vector<float> values;
            for (auto& n : layers.back().neurons) 
                values.emplace_back(n.activation);
            return values;
        }

        void setInputValues(const std::vector<float>& inputValues) {
            if (inputValues.size() != layers.front().neurons.size()) std::cout << "\n\n-----Input values are the wrong size!-----\n\n";
            for (int i = 0; i < inputValues.size(); ++i) layers.front().neurons[i].activation = inputValues[i];
        }

        void calculateNet() {
            for (int i = 0; i < layers.size()-1; ++i) {
                layers[i].calculateNextLayer(layers[i+1].activationFunc);
            }
        }

        void backpropagation(const std::vector<float>& desiredValuesForOutputlayerIndex) {
            calculateDelta(desiredValuesForOutputlayerIndex);
            calculateBackpropChanges();
        }
};
