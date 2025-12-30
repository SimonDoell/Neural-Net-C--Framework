#include <SFML/Graphics.hpp>
#include <./framework.cpp>


//Window Management
const int WIDTH = 1920;
const int HEIGHT = 1010;
int MAX_FRAMES = 60;
bool windowHasFocus;
bool renderData = true;
int amountTrainingPoints = 0;
sf::Font globalFont;

sf::Vector2f dataOffset(WIDTH/4.0f, 0);




// Functions
float vLength(const sf::Vector2f& v) {
    return sqrt(v.x*v.x + v.y*v.y);
}


template<typename T>
void shuffle(std::vector<T> list) {
    for (int i = 0; i < list.size(); ++i) {
        std::swap(list[i], list[rand()%list.size()]);
    }
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


struct DataSet {
    private:
        sf::Vector2f getRandomPos() {
            return sf::Vector2f(rand()%HEIGHT - HEIGHT/2.0f, rand()%HEIGHT - HEIGHT/2.0f);
        }

        float r = 3.0f;

    public:
        std::vector<sf::Vector2f> points;
        std::vector<float> pointValues;

        DataSet() {}

        void render(sf::RenderWindow& window) {
            sf::CircleShape obj(r);
            obj.setOrigin(r, r);
            
            for (int i = 0; i < points.size(); ++i) {
                obj.setPosition(points[i] + dataOffset);
                float val = pointValues[i] * 255.0f;
                obj.setFillColor(sf::Color(val, 0, 255.0f-val, 255));

                window.draw(obj);
            }
        }

        void initRandom(int _amountPoints) {
            for (int i = 0; i < _amountPoints; ++i) {
                points.emplace_back(getRandomPos());

                if (vLength(points.back()) > HEIGHT/3.0f) {
                    pointValues.emplace_back(1.0f);
                } else {
                    pointValues.emplace_back(0.0f);
                }
            }
        }

        void initRects(int _amountPoints) {
            float rectSize = HEIGHT/2.0f;

            for (int i = 0; i < _amountPoints; ++i) {
                points.emplace_back(getRandomPos());
                if (std::abs(int(points.back().y - HEIGHT/2.0f) % int(rectSize)) >= rectSize/2.0f) {
                    pointValues.emplace_back(1.0f);
                } else {
                    pointValues.emplace_back(0.0f);
                }
            }
        }

        void initSinRects(int _amountPoints) {
            float rectSize = HEIGHT/2.0f;

            for (int i = 0; i < _amountPoints; ++i) {
                points.emplace_back(getRandomPos());
                if (std::abs(int(points.back().y - HEIGHT/2.0f) % int(rectSize)) + sin(points.back().x/80.0f)*50.0f >= rectSize/2.0f) {
                    pointValues.emplace_back(1.0f);
                } else {
                    pointValues.emplace_back(0.0f);
                }
            }
        }
};


struct NeuralGrid {
    public:
        ANN& net;
        float gridSize;

        NeuralGrid(ANN& _net, float _gridSize = 5.0f) : net(_net), gridSize(_gridSize) {}

        void render(sf::RenderWindow& window) {
            sf::RectangleShape obj;
            obj.setSize(sf::Vector2f(gridSize, gridSize));
            obj.setOrigin(sf::Vector2f(gridSize/2.0f, gridSize/2.0f));

            for (float x = -HEIGHT/2.0f; x <= HEIGHT/2.0f; x+=gridSize) {
                for (float y = -HEIGHT/2.0f; y <= HEIGHT/2.0f; y+=gridSize) {
                    obj.setPosition(sf::Vector2f(x, y) + dataOffset);
                    
                    net.setInputValues({x/float(HEIGHT) + 0.5f, y/float(HEIGHT) + 0.5f});
                    net.calculateNet();
                    float colorVal = net.getOutputValues().front();
                    colorVal *= 2.0f;
                    colorVal *= 255.0f;
                    colorVal = std::max(std::min(colorVal, 255.0f), 0.0f);

                    obj.setFillColor(sf::Color(colorVal, 0, 255.0f-colorVal, 100));
                    window.draw(obj);
                }
            }
        }
};


struct DisplayText {
    public:
        sf::Text text;
        std::string string;
        sf::Vector2f pos;
        sf::Color color;

        DisplayText(sf::Vector2f _pos, sf::Color _color = sf::Color::White) : pos(_pos), color(_color) {}

        void render(sf::RenderWindow& window) {
            text.setFont(globalFont);
            text.setFillColor(color);
            text.setString(string);
            text.setPosition(pos);
            text.setCharacterSize(40);

            window.draw(text);
        }
};

struct VisualANN {
    public:
        ANN& net;
        sf::Vector2f pos;
        sf::Vector2f size;
        float r = 5.5f;

        std::vector<std::vector<sf::Vector2f>> visualPos;

        VisualANN(ANN& _net, sf::Vector2f _pos, sf::Vector2f _size) : net(_net), pos(_pos), size(_size) {
            visualPos.resize(_net.layersNeuronAmount.size());
            for (int i = 0; i < _net.layersNeuronAmount.size(); ++i)
                visualPos[i].resize(_net.layersNeuronAmount[i]);
        }

        void render(sf::RenderWindow& window) {
            sf::CircleShape obj(r);
            obj.setOrigin(r, r);

            float minX = pos.x;
            float maxX = pos.x + size.x;
            float minY = pos.y;
            float maxY = pos.y + size.y;

            for (int l = 0; l < net.layers.size(); ++l) {
                Layer& currLayer = net.layers[l];
                float xPos = (float(l) / (net.layers.size()-1.0f)) * (maxX-minX) + minX;

                for (int n = 0; n < currLayer.neurons.size(); ++n) {
                    Neuron& currNeuron = currLayer.neurons[n];

                    float yPos;
                    if (currLayer.neurons.size()-1 > 0) {
                        if (n != 0) {
                            yPos = float(n) / (currLayer.neurons.size()-1.0f) * (maxY - minY) + minY;
                        } else {
                            yPos = float(minY);
                            if (minY == 0.0f) {
                                yPos = 1;
                            }
                        }
                    } else {
                        yPos = minY + ((maxY-minY)/2.0f);
                    }

                    float colorVal = currNeuron.bias;
                    colorVal *= 10.0f;
                    colorVal = std::min(std::max(colorVal, 0.0f), 1.0f);
                    colorVal *= 255.0f;

                    obj.setFillColor(sf::Color(colorVal, 0, 255.0f-colorVal, 255));
                    obj.setPosition(sf::Vector2f(xPos, yPos));
                    visualPos[l][n] = sf::Vector2f(xPos, yPos);
                    window.draw(obj);
                }
            }
        }

        void renderNeuronLines(sf::RenderWindow& window) {
            Line line(sf::Vector2f(0, 0), sf::Vector2f(0, 0));

            for (int l = 0; l < visualPos.size()-1; ++l) {
                for (int n = 0; n < visualPos[l].size(); ++n) {
                    for (int nn = 0; nn < visualPos[l+1].size(); ++nn) {
                        line.posA = visualPos[l][n];
                        line.posB = visualPos[l+1][nn];
                        
                        float colorVal = net.layers[l].neurons[n].weights[nn] * 3.0f;
                        colorVal = std::max(std::min(colorVal, 1.0f), 0.0f);
                        colorVal *= 255.0f;

                        line.color = sf::Color(colorVal, 0, 255.0f-colorVal, 255);
                        line.renderOnCanvas(window);
                    }
                }
            }
        }
};


int main() {
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Point Classification ANN", sf::Style::Titlebar | sf::Style::Close);
    window.setFramerateLimit(MAX_FRAMES);
    sf::Event ev;
    sf::ContextSettings settings;
    sf::View view(sf::FloatRect(0, 0, WIDTH, HEIGHT));
    view.zoom(1.05f);
    view.setCenter(sf::Vector2f(0, 0));
    window.setView(view);
    srand(time(0));
    bool pDown = false;
    globalFont.loadFromFile("./Fonts/Roboto/static/Roboto-Light.ttf");

    int amountDataPoints = 10000;

    DataSet data;
    data.initRandom(amountDataPoints);
    std::vector<sf::Vector2f> tempPoints = data.points;

    
    float(*hiddenLayerActivationFunction)(float) = ReLU;
    float(*outputLayerActivationFunction)(float) = Sigmoid;
    ANN net({2, 15, 10, 1}, outputLayerActivationFunction, hiddenLayerActivationFunction);
    net.learningRate = 0.05f;

    NeuralGrid grid(net, 10.0f);


    // Texts
    DisplayText countTrainings(sf::Vector2f(WIDTH/2.0f, HEIGHT/2.0f) * -0.95f, sf::Color::White);
    DisplayText learnRateText(sf::Vector2f(WIDTH/2.0f, HEIGHT/2.0f) * -0.95f + sf::Vector2f(0, 50), sf::Color::Yellow);
    DisplayText neurons(sf::Vector2f(WIDTH/2.0f, HEIGHT/2.0f) * -0.95f + sf::Vector2f(0, 100), sf::Color::Red);

    VisualANN visualNN(net, sf::Vector2f(-WIDTH/2.0f + 50, -150), sf::Vector2f(WIDTH/2.0f - 150, HEIGHT/2.0f + 100));


    while (window.isOpen()) {
        while (window.pollEvent(ev)) {
            if (ev.type == sf::Event::Closed) {window.close(); break;}
            if (ev.type == sf::Event::LostFocus) {windowHasFocus = false;} else if (ev.type == sf::Event::GainedFocus) {windowHasFocus = true;}
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape)) window.close();

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::P)) {
            if (!pDown) {
                renderData = !renderData;
            }
            pDown = true;
        } else {
            pDown = false;
        }

        // Train
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space)) {
            for (int x = 0; x < 1; ++x) {

                shuffle<sf::Vector2f>(tempPoints);

                for (int i = 0; i < tempPoints.size(); ++i) {
                    sf::Vector2f& point = tempPoints[i];
                    net.setInputValues({point.x/float(HEIGHT) + 0.5f, point.y/float(HEIGHT) + 0.5f});
                    net.calculateNet();
                    net.backpropagation({data.pointValues[i]});
                    amountTrainingPoints++;
                }
            }
        }

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up)) {net.learningRate += 0.01f;}
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down)) {net.learningRate -= 0.01f;}
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) {net.learningRate += 0.001f;}
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left)) {net.learningRate -= 0.001f;}



        // Rendering
        window.clear(sf::Color::Black);
        if (renderData) data.render(window);
        grid.render(window);

        
        countTrainings.string = std::to_string(amountTrainingPoints) + " | " + std::to_string(amountDataPoints); countTrainings.render(window);
        learnRateText.string = std::to_string(net.learningRate); learnRateText.string = learnRateText.string.substr(0, 5); learnRateText.render(window);
        

        std::string neuronString = "";
        for (int i = 0; i < net.layersNeuronAmount.size(); ++i) 
            neuronString.append(std::to_string(net.layersNeuronAmount[i]) + (i != net.layersNeuronAmount.size()-1 ? ", " : ""));
        
        neurons.string = neuronString; neurons.render(window);

        visualNN.renderNeuronLines(window);
        visualNN.render(window);
        window.display();
    }
    return 0;
}
