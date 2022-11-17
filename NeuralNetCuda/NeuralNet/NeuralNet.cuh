#ifndef NEURAL_NET_INCLUDED
#define NEURAL_NET_INCLUDED

#include "../header.cuh"

extern int ID;
extern float LEARN_RATE;
extern int BLOCKS;
extern int THREADS;

enum MapType
{
    Input = 0,
    Convolutional = 1,
    Connected = 2,
    Pooling = 3
};

class WeightStorage
{
    public:
        int id;

        int width;
        int height;

        int weightCount;

        vector<float> weights;
        vector<float> weightChanges;

        WeightStorage();
        WeightStorage(int i_width, int i_height, float i_randomBounds);

        void Improve();
};

class NeuralKernel : public WeightStorage
{
    public:
        int kernelSize;

        NeuralKernel(int i_kernelSize, float i_randomBounds);
};

class NeuralMap
{
    public:
        int id;

        MapType mapType;

        bool keepSize = false;

        int mapSize;
        int nodeCount;

        vector<NeuralMap*> previousMaps;

        vector<float> activations;

        vector<float> values;
        vector<float> valueChanges;

        NeuralMap();

        void SetNodeCount(int i_nodeCount);

        void SetValues(vector<float>* i_values);
        vector<float>* GetValues();

        void Correct(vector<float>* i_values);

        void PrintValues();
        string GetMapType();

        virtual void Update() = 0;
        virtual void CalculateChanges() = 0;
        virtual void Improve() = 0;
};

class InputMap : public NeuralMap
{
    public:
        InputMap();
        InputMap(int i_mapSize);

        void Update();
        void CalculateChanges();
        void Improve();
};

class ConvolutionalMap : public NeuralMap
{
    public:
        int kernelSize;
        vector<NeuralKernel> kernels;

        ConvolutionalMap();
        ConvolutionalMap(int i_kernelSize, NeuralMap* i_previousMap);
        ConvolutionalMap(int i_kernelSize, vector<NeuralMap*> i_previousMaps);

        void Update();
        void CalculateChanges();
        void Improve();
};

class PoolingMap : public NeuralMap
{
    public:
        int poolingSize;

        PoolingMap();
        PoolingMap(int i_poolingSize, NeuralMap* i_previousMap);
        PoolingMap(int i_poolingSize, vector<NeuralMap*> i_previousMaps);

        void Update();
        void CalculateChanges();
        void Improve();
};

class ConnectedMap : public NeuralMap
{
    public:
        WeightStorage bias;
        WeightStorage weights;

        int weightCount;

        ConnectedMap();
        ConnectedMap(int i_nodeCount, NeuralMap* i_previousMap);
        ConnectedMap(int i_nodeCount, vector<NeuralMap*> i_previousMaps);

        void Update();
        void CalculateChanges();
        void Improve();
};

class NeuralLayer
{
    public:
        int id;

        NeuralLayer* previousLayer = 0;

        vector<InputMap> inputMaps;
        vector<ConvolutionalMap> convolutionalMaps;
        vector<PoolingMap> poolingMaps;
        vector<ConnectedMap> connectedMaps;

        vector<NeuralMap*> mapPointer;

        MapType layerType;

        int mapCount;

        NeuralLayer();
        NeuralLayer(int i_mapCount, int i_reduction, MapType i_type, NeuralLayer* i_previousLayer);

        void SetValues(vector<vector<float>*> i_values);
        vector<vector<float>*> GetValues();

        void Correct(vector<float>* i_values);

        void PrintValues();

        void Update();
        void CalculateChanges();
        void Improve();
};

class ConvolutionalNetwork
{
    public:
        int id;

        vector<NeuralLayer> layers;

        int layerCount;

        ConvolutionalNetwork(vector<MapType> i_layerType, vector<int> i_mapCount, vector<int> i_reduction);

        void SetValues(vector<vector<float>*> i_values);
        vector<vector<float>*> GetValues(int i_layer);

        void Correct(vector<float>* i_values);
        void Train(vector<vector<float>*> i_trainValues, vector<float>* i_correctvalues);

        void PrintValues();

        void Update();
        void CalculateChanges();
        void Improve();
};

#endif