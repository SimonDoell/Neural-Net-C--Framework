#include <SFML/Graphics.hpp>
#include <list>
#include <vector>
#include <iostream>
#include <cmath>
#include <random>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <chrono>
#include <iomanip>


//Window Management
const int WIDTH = 1920;
const int HEIGHT = 1010;
int MAX_FRAMES = 60;
bool windowHasFocus;

sf::Font globalFont;





//Functions
float detail = 1000;
float getRandomBias() {
    float randNum = ((float(rand() % int(detail)) / detail) - 0.5f) * 2.0f;
    return randNum;
}

sf::Vector2f normalize(const sf::Vector2f& v) {
    float len = sqrt(v.x*v.x + v.y*v.y);
    if (len == 0) return sf::Vector2f(0, 0);
    return v/len;
}


void rotateAroundPoint(sf::Vector2f& v, float rotation) {
    float length = sqrt(v.x * v.x + v.y * v.y);
    v /= length;

    float tempX = v.x * cos(rotation) - v.y * sin(rotation);
    float tempY = v.x * sin(rotation) + v.y * cos(rotation);

    v = sf::Vector2f(tempX, tempY);
    v *= length;
}



float sigmoidDerivative(float x) {
    return (exp(-x) / pow(1 + exp(-x), 2));
}

float reluDerivative(float x) {
    return x <= 0 ? 0 : 1;
}



//Structs
struct Line {
    private:
        sf::VertexArray vertecis;
        float pointR = 5.0f;
        float pointThickness = 2.0f;

    public:
        sf::Vector2f posA, posB;
        sf::Color color = sf::Color::Red;
        float lineWidth = 1.0f;

        Line(sf::Vector2f _posA, sf::Vector2f _posB) : posA(_posA), posB(_posB), vertecis(sf::Quads, 4) {}

        void renderOnCanvas(sf::RenderWindow& window) {
            float dx = posA.x - posB.x;
            float dy = posA.y - posB.y;
            float len = sqrt(dx*dx + dy*dy);

            float rotation = atan2(dy, dx);

            vertecis[0].position = sf::Vector2f(0, lineWidth/2.0f);
            vertecis[1].position = sf::Vector2f(0, -lineWidth/2.0f);
            vertecis[2].position = sf::Vector2f(len, -lineWidth/2.0f);
            vertecis[3].position = sf::Vector2f(len, lineWidth/2.0f);

            rotateAroundPoint(vertecis[0].position, rotation);
            rotateAroundPoint(vertecis[1].position, rotation);
            rotateAroundPoint(vertecis[2].position, rotation);
            rotateAroundPoint(vertecis[3].position, rotation);

            vertecis[0].position += posB;
            vertecis[1].position += posB;
            vertecis[2].position += posB;
            vertecis[3].position += posB;

            vertecis[0].color = color;
            vertecis[1].color = color;
            vertecis[2].color = color;
            vertecis[3].color = color;

            window.draw(vertecis);
        }
};


struct InteractivePad {
    public:
        std::vector<std::vector<float>> values;
        sf::Vector2f pos;
        int size;
        float spacing = 17.0f;
        float displaySize = 15.0f;

        InteractivePad(sf::Vector2f _pos, int _size) : pos(_pos), size(_size) {
            values.resize(size);
            for (int i = 0; i < values.size(); ++i) values[i].resize(size);
        }

        void render(sf::RenderWindow& window) {
            sf::RectangleShape obj;
            obj.setSize(sf::Vector2f(displaySize, displaySize));
            obj.setOrigin(sf::Vector2f(displaySize/2.0f, displaySize/2.0f));

            for (int x = -size/2.0f; x < size/2.0f; ++x) {
                for (int y = -size/2.0f; y < size/2.0f; ++y) {
                    obj.setPosition(pos + sf::Vector2f(x*spacing, y*spacing));

                    float currValue = values[x+size/2.0f][y+size/2.0f]; currValue *= 255.0f;
                    obj.setFillColor(sf::Color(currValue, 255.0f-currValue, 0, 255));

                    window.draw(obj);
                }
            }
        }

        void mouseInteraction(sf::Vector2f mousePos, float addValue) {
            float interactionDistance = 45;

            for (int x = -size/2.0f; x < size/2.0f; ++x) {
                for (int y = -size/2.0f; y < size/2.0f; ++y) {
                    sf::Vector2f rectPos = pos + sf::Vector2f(x*spacing, y*spacing);

                    float dx = mousePos.x - rectPos.x;
                    float dy = mousePos.y - rectPos.y;
                    float distSqr = dx*dx + dy*dy;

                    if (distSqr < interactionDistance*interactionDistance) {
                        float& currValue = values[x+size/2.0f][y+size/2.0f];
                        currValue += addValue;
                        currValue = std::max(std::min(currValue, 1.0f), 0.0f);
                    }
                }
            }
        }

        std::vector<float> getValues() {
            std::vector<float> result = {};

            for (int x = 0; x < values.size(); ++x) {
                for (int y = 0; y < values.size(); ++y) {
                    result.emplace_back(values[y][x]);
                }
            }

            return result;
        }

        void reset() {
            for (int x = 0; x < values.size(); ++x) {
                for (int y = 0; y < values.size(); ++y) {
                    values[x][y] = 0;
                }
            }
        }
};



struct Neuron {
    public:
        float value = 0;
        std::vector<float> connectionWeights;
        float bias;
        sf::Vector2f displayPos;
        float z;  // Pre activation value
        float delta;
        
        
        Neuron(int _nextLayerNeuronAmount) {
            connectionWeights.reserve(_nextLayerNeuronAmount);
            for (int i = 0; i < _nextLayerNeuronAmount; ++i) {
                connectionWeights.emplace_back(getRandomBias());
            }
            bias = getRandomBias();
        }

        sf::Color getColorByValue() {
            float maxVal = 1.0f;  // *Between -1 and 1

            float colorVal = value;
            colorVal = std::min(std::max(-maxVal, colorVal), maxVal);
            colorVal += maxVal;
            colorVal /= maxVal*2.0f;
            colorVal *= 255.0f;

            return sf::Color(colorVal, 0, 255.0f-colorVal, 255.0f);
        }

        sf::Color getColorByConnectionWeights(int connectionIndex) {
            float maxVal = 2.0f;  // *Between -1 and 1

            float colorVal = connectionWeights[connectionIndex];
            colorVal = std::min(std::max(-maxVal, colorVal), maxVal);
            colorVal += maxVal;
            colorVal /= maxVal*2.0f;
            colorVal *= 255.0f;


            return sf::Color(colorVal, 0, 255.0f-colorVal, 255.0f);
        }
};


struct Layer {
    private:
        void resetNextNeuronValues() {
            #pragma omp parallel for
            for (int i = 0; i < nextLayerNeurons.size(); ++i) {
                nextLayerNeurons[i]->value = 0.0f;
            }
        }

        void calcNextLayerBias() {
            #pragma omp parallel for
            for (int i = 0; i < nextLayerNeurons.size(); ++i) {
                nextLayerNeurons[i]->value += nextLayerNeurons[i]->bias;
            }
        }
    
    public:
        std::vector<Neuron*> nextLayerNeurons;
        std::vector<Neuron> neurons;
        int neuronAmount;

        Layer() : neuronAmount(0) {}
        Layer(int _neuronAmount, std::vector<Neuron*> _nextLayerNeurons) : nextLayerNeurons(_nextLayerNeurons), neuronAmount(_neuronAmount) {
            for (int i = 0; i < _neuronAmount; ++i) {
                neurons.emplace_back(Neuron(_nextLayerNeurons.size()));
            }
        }
        
        std::vector<Neuron*> getNeuronsAsPtr() {
            std::vector<Neuron*> pointers = {};
            
            for (int i = 0; i < neurons.size(); ++i) {
                Neuron* currNeuron = &neurons[i];
                pointers.emplace_back(currNeuron);
            }

            return pointers;
        }

        void calculateNextLayer() {
            resetNextNeuronValues();

            for (int i = 0; i < neurons.size(); ++i) {
                float currValue = neurons[i].value;

                for (int x = 0; x < nextLayerNeurons.size(); ++x) {
                    nextLayerNeurons[x]->value += currValue * neurons[i].connectionWeights[x];
                }
            }

            calcNextLayerBias();
        }

        void nextLayerReLU() {
            #pragma omp parallel for
            for (int i = 0; i < nextLayerNeurons.size(); ++i) {
                float& currValue = nextLayerNeurons[i]->value;
                nextLayerNeurons[i]->z = currValue;

                if (currValue < 0) currValue = 0;
            }
        }

        void nextLayerSigmoid() {
            //std::cout << "\n\n\n";
            #pragma omp parallel for
            for (int i = 0; i < nextLayerNeurons.size(); ++i) {
                float& currValue = nextLayerNeurons[i]->value;
                nextLayerNeurons[i]->z = currValue;
                //std::cout << i << " before Sigmoid: " << currValue << "\n";

                currValue = 1.0f / (1.0f + exp(-currValue));
            }
        }
};


struct NeuralNet {
    public:
        std::vector<Layer> layers;
        std::vector<int> layerNeuronAmount;
        float error = 0;

        NeuralNet(std::vector<int> _layerNeuronAmount) : layerNeuronAmount(_layerNeuronAmount) {
            if (layerNeuronAmount.size() < 2) layerNeuronAmount = {16, 10};
            layers.resize(layerNeuronAmount.size());

            layers.back() = Layer(layerNeuronAmount.back(), {});

            for (int i = layerNeuronAmount.size()-2; i >= 0; --i) {
                layers[i] = Layer(layerNeuronAmount[i], layers[i+1].getNeuronsAsPtr());
            }
        };

        void printStructure() {
            for (int i = 0; i < layers.size(); ++i) {
                std::cout << i << ": " << layers[i].neurons.size() << " | Next: " << layers[i].nextLayerNeurons.size() << "\n";
            }
        }

        void printOutputLayer() {
            std::cout << "\n\n";
            for (int i = 0; i < layers.back().neurons.size(); ++i) {
                std::cout << i << ":  " << layers.back().neurons[i].value << "\n";
            }
        }

        void calculateNet() {
            #pragma omp parallel for
            for (int i = 0; i < layers.size()-1; ++i) {
                layers[i].calculateNextLayer();
                if (i != layers.size()-2) {
                    layers[i].nextLayerReLU();
                } else {
                    layers[i].nextLayerSigmoid();
                }
            }
        }

        void calculateError(int desiredIndex, float desiredValue, float otherDesiredValue) {
            error = 0.0f;

            #pragma omp parallel for
            for (int i = 0; i < layers.back().neurons.size(); ++i) {
                float currValue = layers.back().neurons[i].value;


                float diff = 0;
                i == desiredIndex ? diff = currValue-desiredValue : diff = currValue-otherDesiredValue;
                error += diff*diff;
            }
        }

        void render(sf::RenderWindow& window) {
            float xPosOffsetLeft = 800.0f;
            float xPosOffsetRight = 200.0f;
            float xPosFactor = float(WIDTH - xPosOffsetRight - xPosOffsetLeft) / float(layers.size()-1);

            float yPosPadding = 100.0f;

            for (int i = 0; i < layers.size(); ++i) {
                float xPos = float(i)*xPosFactor + xPosOffsetLeft;

                for (int x = 0; x < layers[i].neurons.size(); ++x) {
                    float layerNeuronAmountContraction = layers[i].neurons.size() / 30;

                    float yPosFactor = float(HEIGHT - yPosPadding*2.0f) / float(layers[i].neurons.size()-1);

                    float yPos = float(x)*yPosFactor + yPosPadding;
                    layers[i].neurons[x].displayPos = sf::Vector2f(xPos, yPos);
                }
            }

            float r = 7;
            sf::CircleShape obj(r);
            obj.setOrigin(r, r);
            for (int i = 0; i < layers.size(); ++i) {
                for (int x = 0; x < layers[i].neurons.size(); ++x) {
                    obj.setFillColor(layers[i].neurons[x].getColorByValue());
                    obj.setPosition(layers[i].neurons[x].displayPos);
                    window.draw(obj);
                }
            }
        }

        void renderLines(sf::RenderWindow& window) {
            Line line(sf::Vector2f(0, 0), sf::Vector2f(0, 0));


            for (int i = 0; i < layers.size(); ++i) {
                for (int x = 0; x < layers[i].neurons.size(); ++x) {

                    for (int g = 0; g < layers[i].nextLayerNeurons.size(); ++g) {
                        line.color = layers[i].neurons[x].getColorByConnectionWeights(g);
                        line.posA = layers[i].neurons[x].displayPos;
                        line.posB = layers[i].nextLayerNeurons[g]->displayPos;

                        line.renderOnCanvas(window);
                    }
                }
            }
        }

        void renderOutputNeuronsColumns(sf::RenderWindow& window, bool renderText) {
            sf::RectangleShape obj;
            float objHeight = 20.0f;
            float xOffset = 20.0f;
            float maxLen = 100.0f;
            float textOffset = 20.0f;

            // Finding the highest probability to highlight
            int highestIndex;
            float highestProb = -INFINITY;
            for (int i = 0; i < layers.back().neurons.size(); ++i) {
                Neuron& currNeuron = layers.back().neurons[i];
                if (currNeuron.value > highestProb) {
                    highestProb = currNeuron.value;
                    highestIndex = i;
                }
            }

            for (int i = 0; i < layers.back().neurons.size(); ++i) {
                float length = layers.back().neurons[i].value * maxLen;
                obj.setSize(sf::Vector2f(length, objHeight));
                obj.setPosition(layers.back().neurons[i].displayPos - sf::Vector2f(0, objHeight/2.0f) + sf::Vector2f(xOffset, 0));
                i == highestIndex ? obj.setFillColor(sf::Color::Red) : obj.setFillColor(sf::Color::White);

                if (renderText) {
                    sf::Text numText;
                    numText.setFont(globalFont);
                    i == highestIndex ? numText.setFillColor(sf::Color::Red) : numText.setFillColor(sf::Color::White);
                    numText.setString(std::to_string(i));
                    numText.setCharacterSize(objHeight*2.0f);
                    numText.setPosition(layers.back().neurons[i].displayPos - sf::Vector2f(0, objHeight) + sf::Vector2f(xOffset+maxLen+textOffset, 0));

                    window.draw(numText);
                }

                window.draw(obj);
            }
        }

        void loadValues(const std::vector<float>& values) {
            for (int i = 0; i < layers.front().neurons.size(); ++i) {
                layers.front().neurons[i].value = values[i];
            }
        }



        // Backpropagation
        void trainNet(int desiredIndex, float desiredValue, float otherDesiredValue) {

            // Delta output layer
            for (int i = 0; i < layers.back().neurons.size(); ++i) {
                Neuron& currNeuron = layers.back().neurons[i];

                float targetValue; i == desiredIndex ? targetValue = desiredValue : targetValue = otherDesiredValue;

                currNeuron.delta = (currNeuron.value - targetValue) * sigmoidDerivative(currNeuron.z);
            }


            // Delta Hidden Layers
            for (int i = layers.size()-2; i >= 0; --i) {
                for (int x = 0; x < layers[i].neurons.size(); ++x) {

                    float sum = 0.0f;
                    Neuron& currNeuron = layers[i].neurons[x];

                    for (int j = 0; j < currNeuron.connectionWeights.size(); ++j) {
                        sum += currNeuron.connectionWeights[j] * layers[i].nextLayerNeurons[j]->delta;
                    }

                    currNeuron.delta = sum * reluDerivative(currNeuron.z);
                }
            }


            // Adjusting Weights
            float learningRate = 0.1f;

            for (int l = 0; l < layers.size()-1; ++l) {
                Layer& currLayer = layers[l];
                
                for (int i = 0; i < currLayer.neurons.size(); ++i) {
                    Neuron& currNeuron = currLayer.neurons[i];

                    for (int n = 0; n < currLayer.nextLayerNeurons.size(); ++n) {
                        Neuron* nextNeuron = currLayer.nextLayerNeurons[n];

                        for (int c = 0; c < nextNeuron->connectionWeights.size(); ++c) {
                            nextNeuron->connectionWeights[c] -= learningRate * currNeuron.value * nextNeuron->delta;
                        }
                    }
                }
            }

            // Adjusting bias
            for (int l = 1; l < layers.size(); ++l) {
                for (Neuron& n : layers[l].neurons) {
                    n.bias -= learningRate * n.delta;
                }
            }
        }
};




int main() {
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Neural Net", sf::Style::Titlebar | sf::Style::Close);
    window.setFramerateLimit(MAX_FRAMES);
    sf::Event ev;
    sf::View view(sf::FloatRect(0, 0, WIDTH, HEIGHT));
    view.zoom(1.0f);
    window.setView(view);
    srand(time(0));
    globalFont.loadFromFile("./font/roboto.ttf");
    bool numDown = false;

    
    NeuralNet net({784, 16, 16, 10});


    InteractivePad pad(sf::Vector2f(350.0f, HEIGHT/2.0f), 28);


    net.printStructure();
    net.calculateNet();
    net.printOutputLayer();

    
    


    while (window.isOpen()) {
        while (window.pollEvent(ev)) {
            if (ev.type == sf::Event::Closed) {window.close(); break;}
            if (ev.type == sf::Event::LostFocus) {windowHasFocus = false;} else if (ev.type == sf::Event::GainedFocus) {windowHasFocus = true;}
        }
        // Interactions
        sf::Vector2i mousePosI = sf::Mouse::getPosition(window);
        sf::Vector2f mousePos = window.mapPixelToCoords(mousePosI);

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape)) window.close();

        if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) pad.mouseInteraction(mousePos, 0.05f), net.loadValues(pad.getValues()), net.calculateNet();
        if (sf::Mouse::isButtonPressed(sf::Mouse::Right)) pad.mouseInteraction(mousePos, -0.05f), net.loadValues(pad.getValues()), net.calculateNet();
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::R)) pad.reset(), net.loadValues(pad.getValues()), net.calculateNet();

    
        int numDownCount = 0;
        for (int k = 0; k <= 9; ++k) {
            if (sf::Keyboard::isKeyPressed(static_cast<sf::Keyboard::Key>(sf::Keyboard::Num0 + k))) {
                if (!numDown) {
                    net.trainNet(k, 1, 0);
                    net.calculateNet();
                    net.calculateError(k, 1, 0);
                    std::cout << "\nTrained on: " << k << " | Error: " << net.error << "\n";
                }
                numDown = true;
            } else {
                numDownCount++;
                if (numDownCount >= 10) {
                    numDown = false;
                }
            }
        }




        // Rendering
        window.clear(sf::Color::Black);
        pad.render(window);
        net.renderLines(window);
        net.render(window);
        net.renderOutputNeuronsColumns(window, true);
        window.display();
    }
    return 0;
}
