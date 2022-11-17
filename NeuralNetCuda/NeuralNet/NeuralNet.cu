#include "NeuralNet.cuh"

int ID;
float LEARN_RATE = 0.1f;
int BLOCKS = 1;
int THREADS = 1024;

__device__ float sigmoidCuda(float value)
{
    return (1.0f / (exp(-value) + 1.0f));
}

__device__ float d_sigmoidCuda(float value)
{
    float sig = sigmoidCuda(value);
    float d_sig = sig * (1 - sig);
    return d_sig;
}

__global__ void ResetActivationsZeroCuda(float* activations, int* threadOffset)
{
    int idx = threadIdx.x + *threadOffset;

    activations[idx] = 0;
}

__global__ void ResetActivationsCuda(float* activations, float* bias, int* threadOffset)
{
    int idx = threadIdx.x + *threadOffset;

    activations[idx] = bias[idx];
}

__global__ void UpdateConvolutionalMapCuda(float* activations, float* previousMapValues, float* kernelWeights, int* mapSize, int* kernelSize, int* previousMapSize, int* threadOffset)
{
    int idx = threadIdx.x + *threadOffset;

    int x = idx % *mapSize;
    int y = (idx - x) / *mapSize;

    int mapIndex = idx;

    for (int dY = 0; dY < *kernelSize; dY++)
    {
        for (int dX = 0; dX < *kernelSize; dX++)
        {
            int previousIndex = (x + dX) + *previousMapSize * (y + dY);
            int weightIndex = dX + *kernelSize * dY;

            activations[mapIndex] += previousMapValues[previousIndex] * kernelWeights[weightIndex];
        }
    }
}

__global__ void SetValuesCuda(float* activations, float* values, int* threadOffset)
{
    int idx = threadIdx.x + *threadOffset;

    values[idx] = sigmoidCuda(activations[idx]);
}

__global__ void CalculateChangesConvolutionalMapCuda(float* valueChanges, float* previousMapActivations, float* previousMapValues, float* previousMapValueChanges, float* kernelWeights, float* kernelWeightChanges, int* mapSize, int* kernelSize, int* previousMapSize, bool* pooling, int* threadOffset)
{
    int idx = threadIdx.x + *threadOffset;

    int x = idx % *mapSize;
    int y = (idx - x) / *mapSize;

    int mapIndex = idx;

    for (int dY = 0; dY < *kernelSize; dY++)
    {
        for (int dX = 0; dX < *kernelSize; dX++)
        {
            int previousIndex = (x + dX) + *previousMapSize * (y + dY);
            int weightIndex = dX + *kernelSize * dY;

            kernelWeightChanges[weightIndex] += previousMapValues[previousIndex] * valueChanges[mapIndex];

            if (*pooling)
            {
                previousMapValueChanges[previousIndex] += kernelWeights[weightIndex] * valueChanges[mapIndex];
            }
            else
            {
                previousMapValueChanges[previousIndex] += kernelWeights[weightIndex] * d_sigmoidCuda(previousMapActivations[previousIndex]) * valueChanges[mapIndex];
            }
        }
    }
}

__global__ void UpdatePoolingMapCuda(float* activations, float* values, float* previousMapValues, int* mapSize, int* poolingSize, int* previousMapSize, int* threadOffset)
{
    int idx = threadIdx.x + *threadOffset;

    int x = idx % *mapSize;
    int y = (idx - x) / *mapSize;

    int mapIndex = idx;
    int previousIndex = (x + *previousMapSize * y) * *poolingSize;

    float maximum = previousMapValues[previousIndex];
    for (int dY = 0; dY < *poolingSize; dY++)
    {
        for (int dX = 0; dX < *poolingSize; dX++)
        {
            previousIndex = (x * *poolingSize + dX) + *previousMapSize * (y * *poolingSize + dY);

            float value = previousMapValues[previousIndex];
            maximum = value > maximum ? value : maximum;
        }
    }

    activations[mapIndex] = maximum;
    values[mapIndex] = maximum;
}

__global__ void UpdateConnectedMapCuda(float* activations, float* previousMapValues, float* weights, int* mapSize, int* previousMapIndex, int* previousMapSize, int* previousMapCount, int* threadOffset)
{
    int idx = threadIdx.x + *threadOffset;

    int previousIndex = idx % *previousMapSize;
    int mapIndex = (idx - previousIndex) / *previousMapSize;
    int weightIndex = *previousMapIndex + *previousMapCount * mapIndex + *previousMapCount * *mapSize * previousIndex;

    activations[mapIndex] += previousMapValues[previousIndex] * weights[weightIndex];
}

__global__ void SetBiasCuda(float* valueChanges, float* biasWeightChanges, int* threadOffset)
{
    int idx = threadIdx.x + *threadOffset;

    biasWeightChanges[idx] += valueChanges[idx];
}

__global__ void CalculateChangesConnectedMapCuda(float* valueChanges, float* previousMapActivations, float* previousMapValues, float* previousMapValueChanges, float* weights, float* weightChanges, int* mapSize, int* previousMapIndex, int* previousMapSize, int* previousMapCount, bool* pooling, int* threadOffset)
{
    int idx = threadIdx.x + *threadOffset;

    int previousIndex = idx % *previousMapSize;
    int mapIndex = (idx - previousIndex) / *previousMapSize;
    int weightIndex = *previousMapIndex + *previousMapCount * mapIndex + *previousMapCount * *mapSize * previousIndex;

    weightChanges[weightIndex] += previousMapValues[previousIndex] * valueChanges[mapIndex];

    if (*pooling)
    {
        previousMapValueChanges[previousIndex] += weights[weightIndex] * valueChanges[mapIndex];

    }
    else
    {
        previousMapValueChanges[previousIndex] += weights[weightIndex] * d_sigmoidCuda(previousMapActivations[previousIndex]) * valueChanges[mapIndex];
    }
    
}

float sigmoid(float value)
{
    return (1.0f / (exp(-value) + 1.0f));
}

float d_sigmoid(float value)
{
    float sig = sigmoid(value);
    float d_sig = sig*(1-sig);
    return d_sig;
}

WeightStorage::WeightStorage()
{
}

WeightStorage::WeightStorage(int i_width, int i_height, float i_randomBounds)
{
    id = ID++;

    width = i_width;
    height = i_height;

    weightCount = width * height;

    for(int i = 0; i < weightCount; i++)
    {
        weights.push_back(static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (2*i_randomBounds))) - i_randomBounds);
        weightChanges.push_back(0.0f);
    }
}

void WeightStorage::Improve()
{
    for(int i = 0; i < weightCount; i++)
    {
        weights[i] -= LEARN_RATE * weightChanges[i];
        weightChanges[i] = 0;
    }
}

NeuralKernel::NeuralKernel(int i_kernelSize, float i_randomBounds) : WeightStorage(i_kernelSize, i_kernelSize, i_randomBounds)
{
    kernelSize = i_kernelSize;
}

NeuralMap::NeuralMap()
{
}

void NeuralMap::SetNodeCount(int i_nodeCount)
{
    nodeCount = i_nodeCount;

    activations = vector<float>(nodeCount);
    values = vector<float>(nodeCount);
    valueChanges = vector<float>(nodeCount);
}

void NeuralMap::SetValues(vector<float>* i_values)
{
    for(int i = 0; i < i_values->size(); i++)
    {
        values[i] = (*i_values)[i];
    }
}

vector<float>* NeuralMap::GetValues()
{
    return &values;
}

void NeuralMap::Correct(vector<float>* i_values)
{    
    for(int i = 0; i < (*i_values).size(); i ++)
    {
        if((*i_values)[i] != - 1)
        {
            valueChanges[i] = 2*(values[i] - (*i_values)[i]);
        }
    }   
}

void NeuralMap::PrintValues()
{
    cout << GetMapType() << " (id: " << id << ", size: " << mapSize <<  ") ";

    if(previousMaps.size() != 0)
    {
        cout << "[input id: ";
        for(int i = 0; i < previousMaps.size()-1; i ++)
        {
            cout << previousMaps[i]->id << ", ";
        }
        cout << previousMaps[previousMaps.size() - 1]->id << "]";
    }

    cout << endl;

    for(int i = 0; i < nodeCount; i++)
    {
        if(((i+1) % mapSize) == 0)
        {
            cout << fixed << setprecision(2) << values[i] << endl;
        }
        else
        {
            cout << fixed << setprecision(2)  << values[i] << " | ";
        }
    }
}

string NeuralMap::GetMapType()
{
    switch (mapType)
    {
        case Input:
            return "Input Map";
        case Convolutional:
            return "Convolutional Map";
        case Pooling:
            return "Pooling Map";
        case Connected:
            return "Connected Map";
        default:
            return "Other Map";
    }
}

InputMap::InputMap()
{
}

InputMap::InputMap(int i_mapSize)
{
    id = ID++;

    mapType = Input;
    mapSize = i_mapSize;

    SetNodeCount(i_mapSize * i_mapSize);
}

void InputMap::Update()
{
}

void InputMap::CalculateChanges()
{
}

void InputMap::Improve()
{
}

ConvolutionalMap::ConvolutionalMap()
{      
}

ConvolutionalMap::ConvolutionalMap(int i_kernelSize, NeuralMap* i_previousMap)
{
    id = ID++;

    mapType = Convolutional;

    kernelSize = i_kernelSize;

    previousMaps = {i_previousMap};
    mapSize = previousMaps[0]->mapSize - kernelSize + 1;

    SetNodeCount(mapSize * mapSize);
            
    for(int i = 0; i < previousMaps.size(); i ++)
    {
        kernels.push_back(NeuralKernel(kernelSize, 3.0f));
    }
}

ConvolutionalMap::ConvolutionalMap(int i_kernelSize, vector<NeuralMap*> i_previousMaps)
{      
    id = ID++;

    mapType = Convolutional;

    kernelSize = i_kernelSize;

    previousMaps = i_previousMaps;
    mapSize = previousMaps[0]->mapSize - kernelSize + 1;

    SetNodeCount(mapSize * mapSize);
            
    for(int i = 0; i < previousMaps.size(); i ++)
    {
        kernels.push_back(NeuralKernel(kernelSize, 3.0f));
    }
}

void ConvolutionalMap::Update()
{
    int previousMapCount = previousMaps.size();
    int previousMapSize = previousMaps[0]->mapSize;

    cudaSetDevice(0);

    int* dev_threadOffset = 0;
    
    float* dev_activations = 0;
    float* dev_values = 0;

    float* dev_previousMapValues = 0;
    float* dev_kernelWeights = 0;

    int* dev_mapSize = 0;
    int* dev_previousMapSize = 0;
    int* dev_kernelSize = 0;

    cudaMalloc((void**)&dev_threadOffset, 1 * sizeof(int));

    cudaMalloc((void**)&dev_activations, mapSize * mapSize * sizeof(float));
    cudaMalloc((void**)&dev_values, mapSize * mapSize * sizeof(float));

    cudaMalloc((void**)&dev_mapSize, 1 * sizeof(int));
    cudaMalloc((void**)&dev_previousMapSize, 1 * sizeof(int));
    cudaMalloc((void**)&dev_kernelSize, 1 * sizeof(int));

    cudaMalloc((void**)&dev_previousMapValues, previousMapSize * previousMapSize * sizeof(float));
    cudaMalloc((void**)&dev_kernelWeights, kernelSize * kernelSize * sizeof(float));

    cudaMemcpy(dev_activations, &activations[0], mapSize * mapSize * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(dev_mapSize, &mapSize, 1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_previousMapSize, &previousMapSize, 1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_kernelSize, &kernelSize, 1 * sizeof(int), cudaMemcpyHostToDevice);

    int allThreads = mapSize * mapSize;

    for (int i = 0; i < allThreads; i += THREADS)
    {
        int threadOffset = i;

        cudaMemcpy(dev_threadOffset, &threadOffset, 1 * sizeof(int), cudaMemcpyHostToDevice);

        ResetActivationsZeroCuda << <BLOCKS, min(THREADS, allThreads - i) >> > (dev_activations, dev_threadOffset);
    }

    cudaDeviceSynchronize();
    
    for(int i = 0; i < previousMapCount; i ++)
    {
        NeuralMap* previousMap = previousMaps[i];

        NeuralKernel* kernel = &kernels[i];

        cudaMemcpy(dev_previousMapValues, &previousMap->values[0], previousMapSize * previousMapSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_kernelWeights, &kernel->weights[0], kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);
        
        for (int j = 0; j < allThreads; j += THREADS)
        {
            int threadOffset = j;
            
            cudaMemcpy(dev_threadOffset, &threadOffset, 1 * sizeof(int), cudaMemcpyHostToDevice);

            UpdateConvolutionalMapCuda << <BLOCKS, min(THREADS, allThreads - j) >> > (dev_activations, dev_previousMapValues, dev_kernelWeights, dev_mapSize, dev_kernelSize, dev_previousMapSize, dev_threadOffset);
        }
    }

    cudaDeviceSynchronize();

    for (int i = 0; i < allThreads; i += THREADS)
    {
        int threadOffset = i;

        cudaMemcpy(dev_threadOffset, &threadOffset, 1 * sizeof(int), cudaMemcpyHostToDevice);

        SetValuesCuda << <BLOCKS, min(THREADS, allThreads - i) >> > (dev_activations, dev_values, dev_threadOffset);
    }

    cudaDeviceSynchronize();

    cudaMemcpy(&activations[0], dev_activations, mapSize * mapSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&values[0], dev_values, mapSize * mapSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_threadOffset);

    cudaFree(dev_activations);
    cudaFree(dev_values);

    cudaFree(dev_previousMapValues);
    cudaFree(dev_kernelWeights);

    cudaFree(dev_mapSize);
    cudaFree(dev_previousMapSize);
    cudaFree(dev_kernelSize);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ConvolutionalMap::Update failed: %s\n", cudaGetErrorString(cudaStatus));
    }
}

void ConvolutionalMap::CalculateChanges()
{
    int previousMapCount = previousMaps.size();
    int previousMapSize = previousMaps[0]->mapSize;

    bool pooling = previousMaps[0]->mapType == Pooling;

    cudaSetDevice(0);

    int* dev_threadOffset = 0;

    float* dev_valueChanges = 0;
    float* dev_previousMapActivations = 0;
    float* dev_previousMapValues = 0;
    float* dev_previousMapValueChanges = 0;
    float* dev_kernelWeights = 0;
    float* dev_kernelWeightChanges = 0;

    int* dev_kernelSize = 0;
    int* dev_previousMapSize = 0;
    int* dev_mapSize = 0;

    bool* dev_pooling = false;

    cudaMalloc(&dev_threadOffset, 1 * sizeof(int));
    
    cudaMalloc((void**)&dev_valueChanges, mapSize * mapSize * sizeof(float));

    cudaMalloc((void**)&dev_previousMapActivations, previousMapSize * previousMapSize * sizeof(float));
    cudaMalloc((void**)&dev_previousMapValues, previousMapSize * previousMapSize * sizeof(float));
    cudaMalloc((void**)&dev_previousMapValueChanges, previousMapSize * previousMapSize * sizeof(float));
    cudaMalloc((void**)&dev_kernelWeights, kernelSize * kernelSize * sizeof(float));
    cudaMalloc((void**)&dev_kernelWeightChanges, kernelSize * kernelSize * sizeof(float));

    cudaMalloc((void**)&dev_previousMapSize, 1 * sizeof(int));
    cudaMalloc((void**)&dev_mapSize, 1 * sizeof(int));
    cudaMalloc((void**)&dev_kernelSize, 1 * sizeof(int));

    cudaMalloc((void**)&dev_pooling, 1 * sizeof(bool));

    cudaMemcpy(dev_valueChanges, &valueChanges[0], mapSize * mapSize * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(dev_mapSize, &mapSize, 1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_previousMapSize, &previousMapSize, 1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_kernelSize, &kernelSize, 1 * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(dev_pooling, &pooling, 1 * sizeof(bool), cudaMemcpyHostToDevice);

    for(int i = 0; i < previousMapCount; i ++)
    {
        NeuralMap* previousMap = previousMaps[i];

        NeuralKernel* kernel = &kernels[i];

        cudaMemcpy(dev_previousMapActivations, &previousMap->activations[0], previousMapSize * previousMapSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_previousMapValues, &previousMap->values[0], previousMapSize * previousMapSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_previousMapValueChanges, &previousMap->valueChanges[0], previousMapSize * previousMapSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_kernelWeights, &kernel->weights[0], kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_kernelWeightChanges, &kernel->weightChanges[0], kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);
        
        int allThreads = mapSize * mapSize;

        for (int j = 0; j < allThreads; j += THREADS)
        {
            int threadOffset = j;

            cudaMemcpy(dev_threadOffset, &threadOffset, 1 * sizeof(int), cudaMemcpyHostToDevice);

            CalculateChangesConvolutionalMapCuda << <BLOCKS, min(THREADS, allThreads - j) >> > (dev_valueChanges, dev_previousMapActivations, dev_previousMapValues, dev_previousMapValueChanges, dev_kernelWeights, dev_kernelWeightChanges, dev_mapSize, dev_kernelSize, dev_previousMapSize, dev_pooling, dev_threadOffset);
        }

        cudaDeviceSynchronize();

        cudaMemcpy(&kernel->weightChanges[0], dev_kernelWeightChanges, kernelSize * kernelSize * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&previousMap->valueChanges[0], dev_previousMapValueChanges, previousMapSize * previousMapSize * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaFree(dev_threadOffset);

    cudaFree(dev_valueChanges);
    cudaFree(dev_previousMapActivations);
    cudaFree(dev_previousMapValues);
    cudaFree(dev_previousMapValueChanges);
    cudaFree(dev_kernelWeights);
    cudaFree(dev_kernelWeightChanges);

    cudaFree(dev_kernelSize);
    cudaFree(dev_previousMapSize);
    cudaFree(dev_mapSize);

    cudaFree(dev_pooling);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ConvolutionalMap::CalculateChanges failed: %s\n", cudaGetErrorString(cudaStatus));
    }
}

void ConvolutionalMap::Improve()
{
    for(int i = 0; i < kernels.size(); i ++)
    {
        kernels[i].Improve();
    }

    for(int i = 0; i < nodeCount; i++)
    {
        valueChanges[i] = 0;
    }     
}

PoolingMap::PoolingMap()
{
}

PoolingMap::PoolingMap(int i_poolingSize, NeuralMap* i_previousMap)
{
    id = ID++;

    mapType = Pooling;

    previousMaps = {i_previousMap};

    poolingSize = i_poolingSize;
    mapSize = previousMaps[0]->mapSize / poolingSize;

    SetNodeCount(mapSize * mapSize);
}

PoolingMap::PoolingMap(int i_poolingSize, vector<NeuralMap*> i_previousMaps)
{
    id = ID++;

    mapType = Pooling;

    previousMaps = i_previousMaps;

    poolingSize = i_poolingSize;
    mapSize = previousMaps[0]->mapSize / poolingSize;

    SetNodeCount(mapSize * mapSize);
}

void PoolingMap::Update()
{
    int previousMapCount = previousMaps.size();
    int previousMapSize = previousMaps[0]->mapSize;

    cudaSetDevice(0);

    int* dev_threadOffset = 0;

    float* dev_activations = 0;
    float* dev_values = 0;
    float* dev_previousMapValues = 0;

    int* dev_poolingSize = 0;
    int* dev_previousMapSize = 0;
    int* dev_mapSize = 0;

    cudaMalloc(&dev_threadOffset, 1 * sizeof(int));

    cudaMalloc((void**)&dev_activations, mapSize * mapSize * sizeof(float));
    cudaMalloc((void**)&dev_values, mapSize * mapSize * sizeof(float));
    cudaMalloc((void**)&dev_previousMapValues, previousMapSize * previousMapSize * sizeof(float));

    cudaMalloc((void**)&dev_previousMapSize, 1 * sizeof(int));
    cudaMalloc((void**)&dev_mapSize, 1 * sizeof(int));
    cudaMalloc((void**)&dev_poolingSize, 1 * sizeof(int));

    cudaMemcpy(dev_mapSize, &mapSize, 1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_previousMapSize, &previousMapSize, 1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_poolingSize, &poolingSize, 1 * sizeof(int), cudaMemcpyHostToDevice);

    for(int i = 0; i < previousMapCount; i ++)
    {
        NeuralMap* previousMap = previousMaps[i];

        cudaMemcpy(dev_previousMapValues, &previousMap->values[0], previousMapSize * previousMapSize * sizeof(float), cudaMemcpyHostToDevice);

        int allThreads = mapSize * mapSize;

        for (int j = 0; j < allThreads; j += THREADS)
        {
            int threadOffset = j;

            cudaMemcpy(dev_threadOffset, &threadOffset, 1 * sizeof(int), cudaMemcpyHostToDevice);

            UpdatePoolingMapCuda << <BLOCKS, min(THREADS, allThreads - j) >> > (dev_activations, dev_values, dev_previousMapValues, dev_mapSize, dev_poolingSize, dev_previousMapSize, dev_threadOffset);
        }

        cudaDeviceSynchronize();
    }

    cudaMemcpy(&activations[0], dev_activations, mapSize * mapSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&values[0], dev_values, mapSize * mapSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_values);
    cudaFree(dev_previousMapValues);

    cudaFree(dev_previousMapSize);
    cudaFree(dev_mapSize);
    cudaFree(dev_poolingSize);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "PoolingMap::Update failed: %s\n", cudaGetErrorString(cudaStatus));
    }
}

void PoolingMap::CalculateChanges()
{
    for(int i = 0; i < previousMaps.size(); i ++)
        {
        NeuralMap* previousMap = previousMaps[i];
        int previousMapSize = previousMap->mapSize;

        for(int y = 0; y < mapSize; y ++)
        {
            for(int x = 0; x < mapSize; x ++)
            {
                int mapIndex = x + mapSize * y;
                int previousIndex = (x + previousMapSize * y) * poolingSize;
                
                float maximum = previousMap->values[previousIndex];
                int maximumIndex = previousIndex;

                for(int dY = 0; dY < poolingSize; dY ++)
                {
                    for(int dX = 0; dX < poolingSize; dX ++)
                    {
                        float value = previousMap->values[previousIndex];

                        maximumIndex = value > maximum ? previousIndex : maximumIndex;
                        maximum = value > maximum ? value : maximum;
                    }
                }

                previousMap->valueChanges[maximumIndex] += valueChanges[mapIndex] * d_sigmoid(previousMap->activations[maximumIndex]);
            }
        }
    }
}

void PoolingMap::Improve()
{
    for(int i = 0; i < nodeCount; i++)
    {
        valueChanges[i] = 0;
    }
}

ConnectedMap::ConnectedMap()
{
}

ConnectedMap::ConnectedMap(int i_nodeCount, NeuralMap* i_previousMap)
{
    id = ID++;

    mapType = Connected;
    mapSize = i_nodeCount;

    SetNodeCount(i_nodeCount);

    previousMaps = {i_previousMap};

    bias = WeightStorage(mapSize, 1, 1.0f);
    weights = WeightStorage(mapSize, previousMaps[0]->mapSize, 1.0f);

    weightCount = weights.weightCount;
}

ConnectedMap::ConnectedMap(int i_nodeCount, vector<NeuralMap*> i_previousMaps)
{
    id = ID++;

    mapType = Connected;
    mapSize = i_nodeCount;

    SetNodeCount(i_nodeCount);

    previousMaps = i_previousMaps;

    bias = WeightStorage(mapSize, 1, 1.0f);
    weights = WeightStorage(mapSize, previousMaps.size() * previousMaps[0]->mapSize, 1.0f);

    weightCount = weights.weightCount;
}

void ConnectedMap::Update()
{
    int previousMapCount = previousMaps.size();
    int previousMapSize = previousMaps[0]->mapSize;

    cudaSetDevice(0);

    int* dev_threadOffset = 0;

    float* dev_activations = 0;
    float* dev_values = 0;
    float* dev_bias = 0;
    float* dev_weights = 0;
    float* dev_previousMapValues = 0;

    int* dev_mapSize = 0;
    int* dev_previousMapCount = 0;

    int* dev_previousMapIndex = 0;
    int* dev_previousMapSize = 0;

    cudaMalloc((void**)&dev_threadOffset, 1 * sizeof(int));

    cudaMalloc((void**)&dev_activations, mapSize * sizeof(float));
    cudaMalloc((void**)&dev_values, mapSize * sizeof(float));
    cudaMalloc((void**)&dev_bias, mapSize * sizeof(float));
    cudaMalloc((void**)&dev_weights, mapSize * previousMapSize * previousMapCount * sizeof(float));
    cudaMalloc((void**)&dev_previousMapValues, previousMapSize * sizeof(float));

    cudaMalloc((void**)&dev_mapSize, 1 * sizeof(int));
    cudaMalloc((void**)&dev_previousMapCount, 1 * sizeof(int));

    cudaMalloc((void**)&dev_previousMapIndex, 1 * sizeof(int));
    cudaMalloc((void**)&dev_previousMapSize, 1 * sizeof(int));

    cudaMemcpy(dev_activations, &activations[0], mapSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_bias, &bias.weights[0], mapSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_weights, &weights.weights[0], mapSize * previousMapSize * previousMapCount * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(dev_mapSize, &mapSize, 1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_previousMapSize, &previousMapSize, 1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_previousMapCount, &previousMapCount, 1 * sizeof(int), cudaMemcpyHostToDevice);

    int allThreads = mapSize;

    for (int i = 0; i < allThreads; i += THREADS)
    {
        int threadOffset = i;

        cudaMemcpy(dev_threadOffset, &threadOffset, 1 * sizeof(int), cudaMemcpyHostToDevice);

        ResetActivationsCuda << <BLOCKS, min(THREADS, allThreads - i) >> > (dev_activations, dev_bias, dev_threadOffset);
    }

    cudaDeviceSynchronize();

    for(int i = 0; i < previousMaps.size(); i ++)
    {
        int previousMapIndex = i;
        
        NeuralMap* previousMap = previousMaps[previousMapIndex];

        cudaMemcpy(dev_previousMapValues, &previousMap->values[0], previousMapSize * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemcpy(dev_previousMapIndex, &previousMapIndex, 1 * sizeof(int), cudaMemcpyHostToDevice);

        allThreads = mapSize * previousMapSize;

        for (int i = 0; i < allThreads; i += THREADS)
        {
            int threadOffset = i;

            cudaMemcpy(dev_threadOffset, &threadOffset, 1 * sizeof(int), cudaMemcpyHostToDevice);

            UpdateConnectedMapCuda << <BLOCKS, min(THREADS, allThreads - i) >> > (dev_activations, dev_previousMapValues, dev_weights, dev_mapSize, dev_previousMapIndex, dev_previousMapSize, dev_previousMapCount, dev_threadOffset);
        }
    }

    cudaDeviceSynchronize();

    allThreads = mapSize;

    for (int i = 0; i < allThreads; i += THREADS)
    {
        int threadOffset = i;

        cudaMemcpy(dev_threadOffset, &threadOffset, 1 * sizeof(int), cudaMemcpyHostToDevice);

        SetValuesCuda << <BLOCKS, min(THREADS, allThreads - i) >> > (dev_activations, dev_values, dev_threadOffset);
    }

    cudaDeviceSynchronize();

    cudaMemcpy(&activations[0], dev_activations, mapSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&values[0], dev_values, mapSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_threadOffset);
    
    cudaFree(dev_activations);
    cudaFree(dev_values);

    cudaFree(dev_previousMapValues);
    cudaFree(dev_weights);

    cudaFree(dev_previousMapIndex);
    cudaFree(dev_previousMapSize);
    cudaFree(dev_bias);

    cudaFree(dev_mapSize);
    cudaFree(dev_previousMapCount);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ConnectedMap::Update failed: %s\n", cudaGetErrorString(cudaStatus));
    }
}

void ConnectedMap::CalculateChanges()
{
    int previousMapCount = previousMaps.size();
    int previousMapSize = previousMaps[0]->mapSize;

    bool pooling = previousMaps[0]->mapType == Pooling;

    cudaSetDevice(0);

    int* dev_threadOffset = 0;

    float* dev_activations = 0;
    float* dev_valueChanges = 0;
    float* dev_bias = 0;
    float* dev_biasChanges = 0;
    float* dev_weights = 0;
    float* dev_weightChanges = 0;
    float* dev_previousMapActivations = 0;
    float* dev_previousMapValues = 0;
    float* dev_previousMapValueChanges = 0;

    int* dev_mapSize = 0;
    int* dev_previousMapCount = 0;

    int* dev_previousMapIndex = 0;
    int* dev_previousMapSize = 0;

    bool* dev_pooling = 0;

    cudaMalloc((void**)&dev_threadOffset, 1 * sizeof(int));

    cudaMalloc((void**)&dev_activations, mapSize * sizeof(float));
    cudaMalloc((void**)&dev_valueChanges, mapSize * sizeof(float));
    cudaMalloc((void**)&dev_bias, mapSize * sizeof(float));
    cudaMalloc((void**)&dev_biasChanges, mapSize * sizeof(float));
    cudaMalloc((void**)&dev_weights, mapSize * previousMapSize * previousMaps.size() * sizeof(float));
    cudaMalloc((void**)&dev_weightChanges, mapSize * previousMapSize * previousMaps.size() * sizeof(float));
    cudaMalloc((void**)&dev_previousMapActivations, previousMapSize * sizeof(float));
    cudaMalloc((void**)&dev_previousMapValues, previousMapSize * sizeof(float));
    cudaMalloc((void**)&dev_previousMapValueChanges, previousMapSize * sizeof(float));

    cudaMalloc((void**)&dev_mapSize, 1 * sizeof(int));
    cudaMalloc((void**)&dev_previousMapCount, 1 * sizeof(int));

    cudaMalloc((void**)&dev_previousMapIndex, 1 * sizeof(int));
    cudaMalloc((void**)&dev_previousMapSize, 1 * sizeof(int));

    cudaMalloc((void**)&dev_pooling, 1 * sizeof(bool));

    cudaMemcpy(dev_activations, &activations[0], mapSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_valueChanges, &valueChanges[0], mapSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_bias, &bias.weights[0], mapSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_biasChanges, &bias.weightChanges[0], mapSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_weights, &weights.weights[0], mapSize * previousMapSize * previousMapCount * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_weightChanges, &weights.weightChanges[0], mapSize * previousMapSize * previousMapCount * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(dev_mapSize, &mapSize, 1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_previousMapSize, &previousMapSize, 1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_previousMapCount, &previousMapCount, 1 * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(dev_pooling, &pooling, 1 * sizeof(bool), cudaMemcpyHostToDevice);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ConnectedMap::CalculateChanges SetUp failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    int allThreads = mapSize;

    for (int i = 0; i < allThreads; i += THREADS)
    {
        int threadOffset = i;

        cudaMemcpy(dev_threadOffset, &threadOffset, 1 * sizeof(int), cudaMemcpyHostToDevice);

        SetBiasCuda << <BLOCKS, min(THREADS, allThreads - i) >> > (dev_valueChanges, dev_biasChanges, dev_threadOffset);
    }

    cudaDeviceSynchronize();

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ConnectedMap::CalculateChanges Bias failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    for(int i = 0; i < previousMapCount; i ++)
    {
        int previousMapIndex = i;

        NeuralMap* previousMap = previousMaps[previousMapIndex];

        cudaMemcpy(dev_previousMapActivations, &previousMap->activations[0], previousMapSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_previousMapValues, &previousMap->values[0], previousMapSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_previousMapValueChanges, &previousMap->valueChanges[0], previousMapSize * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemcpy(dev_previousMapIndex, &previousMapIndex, 1 * sizeof(int), cudaMemcpyHostToDevice);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "ConnectedMap::CalculateChanges Changes Memcpy failed: %s\n", cudaGetErrorString(cudaStatus));
        }

        allThreads = mapSize * previousMapSize;

        for (int j = 0; j < allThreads; j += THREADS)
        {
            int threadOffset = j;

            cudaMemcpy(dev_threadOffset, &threadOffset, 1 * sizeof(int), cudaMemcpyHostToDevice);

            CalculateChangesConnectedMapCuda << <BLOCKS, min(THREADS, allThreads - j) >> > (dev_valueChanges, dev_previousMapActivations, dev_previousMapValues, dev_previousMapValueChanges, dev_weights, dev_weightChanges, dev_mapSize, dev_previousMapIndex, dev_previousMapSize, dev_previousMapCount, dev_pooling, dev_threadOffset);
        }

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "ConnectedMap::CalculateChanges Changes Run failed: %s\n", cudaGetErrorString(cudaStatus));
        }

        cudaDeviceSynchronize();

        cudaMemcpy(&previousMap->valueChanges[0], dev_previousMapValueChanges, previousMapSize * sizeof(float), cudaMemcpyDeviceToHost);
    
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "ConnectedMap::CalculateChanges Changes Extract failed: %s\n", cudaGetErrorString(cudaStatus));
        }
    }

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ConnectedMap::CalculateChanges Changes failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    cudaMemcpy(&bias.weightChanges[0], dev_biasChanges, mapSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&weights.weightChanges[0], dev_weightChanges, mapSize * previousMapSize * previousMaps.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_threadOffset);

    cudaFree(dev_activations);
    cudaFree(dev_valueChanges);

    cudaFree(dev_previousMapActivations);
    cudaFree(dev_previousMapValues);
    cudaFree(dev_previousMapValueChanges);
    cudaFree(dev_weights);
    cudaFree(dev_weightChanges);

    cudaFree(dev_previousMapIndex);
    cudaFree(dev_previousMapSize);
    cudaFree(dev_bias);
    cudaFree(dev_biasChanges);

    cudaFree(dev_mapSize);
    cudaFree(dev_previousMapCount);

    cudaFree(dev_pooling);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "ConnectedMap::CalculateChanges Free failed: %s\n", cudaGetErrorString(cudaStatus));
    }
}

void ConnectedMap::Improve()
{
    bias.Improve();
    weights.Improve();

    for(int i = 0; i < mapSize; i ++)
    {
        valueChanges[i] = 0;
    }
}

NeuralLayer::NeuralLayer()
{
}

NeuralLayer::NeuralLayer(int i_mapCount, int i_reduction, MapType i_type, NeuralLayer* i_previousLayer)
{
    id = ID++;

    layerType = i_type;

    mapCount = i_mapCount;

    if(layerType == Input)
    {
        inputMaps = vector<InputMap> (mapCount);

        for(int i = 0; i < mapCount; i ++)
        {
            inputMaps[i] = InputMap(i_reduction);
            mapPointer.push_back(&inputMaps[i]);
        }
    }
    else if(layerType == Convolutional)
    {
        convolutionalMaps = vector<ConvolutionalMap> (mapCount);

        for(int i = 0; i < mapCount; i ++)
        {
            convolutionalMaps[i] = ConvolutionalMap(i_reduction, i_previousLayer->mapPointer);
            mapPointer.push_back(&convolutionalMaps[i]);
        }
    }
    else if(layerType == Pooling)
    {
        poolingMaps = vector<PoolingMap> (mapCount);

        for(int i = 0; i < mapCount; i ++)
        {
            poolingMaps[i] = PoolingMap(i_reduction, i_previousLayer->mapPointer[i]);
            mapPointer.push_back(&poolingMaps[i]);
        }
    }
    else if(layerType == Connected)
    {
        connectedMaps = vector<ConnectedMap> (mapCount);

        for(int i = 0; i < mapCount; i ++)
        {
            connectedMaps[i] = ConnectedMap(i_reduction, i_previousLayer->mapPointer[i]);
            mapPointer.push_back(&connectedMaps[i]);
        }
    }
}

void NeuralLayer::SetValues(vector<vector<float>*> i_values)
{
    for(int i = 0; i < mapCount; i ++)
    {
        mapPointer[i]->SetValues(i_values[i]);
    }
}

vector<vector<float>*> NeuralLayer::GetValues()
{
    vector<vector<float>*> results;
    for(int i = 0; i < mapCount; i ++)
    {
        results.push_back(mapPointer[i]->GetValues());
    }
    return results;
}

void NeuralLayer::Correct(vector<float>* i_values)
{    
    mapPointer[0]->Correct(i_values);  
}

void NeuralLayer::PrintValues()
{
    cout << "Neural Layer (id: " << id << ")" << endl;
    for(int i = 0; i < mapCount; i ++)
    {
        mapPointer[i]->PrintValues();
    }
}

void NeuralLayer::Update()
{
    for(int i = 0; i < mapCount; i ++)
    {
        //cout << "       Update map " << (i + 1) << " / " << mapCount << endl;
        mapPointer[i]->Update();
    }
}

void NeuralLayer::CalculateChanges()
{
    for(int i = 0; i < mapCount; i ++)
    {
        //cout << "       Calculate map " << (i + 1) << " / " << mapCount << endl;
        mapPointer[i]->CalculateChanges();
    }
}

void NeuralLayer::Improve()
{
    for(int i = 0; i < mapCount; i ++)
    {
        mapPointer[i]->Improve();
    }
}

ConvolutionalNetwork::ConvolutionalNetwork(vector<MapType> i_layerType, vector<int> i_mapCount, vector<int> i_reduction)
{
    id = ID++;

    layerCount = i_layerType.size();

    layers = vector<NeuralLayer> (layerCount);

    for(int i = 0; i < layerCount; i ++)
    {
        layers[i] = NeuralLayer(i_mapCount[i], i_reduction[i], i_layerType[i], &layers[max(i - 1,0)]);
    }
}


void ConvolutionalNetwork::SetValues(vector<vector<float>*> i_values)
{
    layers[0].SetValues(i_values);
}

vector<vector<float>*> ConvolutionalNetwork::GetValues(int i_layer)
{
    return layers[i_layer].GetValues();
}

void ConvolutionalNetwork::Correct(vector<float>* i_values)
{    
    layers[layerCount - 1].Correct(i_values);  
}

void ConvolutionalNetwork::Train(vector<vector<float>*> i_trainValues, vector<float>* i_correctvalues)
{    
    //cout << "Set net input" << endl;
    SetValues(i_trainValues);

    //cout << "Update net" << endl;
    Update();

    //cout << "Get correct output" << endl;
    Correct(i_correctvalues);

    //cout << "Calculate net changes" << endl;
    CalculateChanges();
}

void ConvolutionalNetwork::PrintValues()
{
    cout << "Convolutional Network (id: " << id << ")" << endl;
    for(int i = 0; i < layerCount; i ++)
    {
        layers[i].PrintValues();
    }
}

void ConvolutionalNetwork::Update()
{
    for(int i = 0; i < layerCount; i ++)
    {
        //cout << "   Update layer " << (i + 1) << " / " << layerCount << endl;
        layers[i].Update();
    }
}

void ConvolutionalNetwork::CalculateChanges()
{
    for(int i = layerCount - 1; i >= 0; i --)
    {
        //cout << "   Calculate layer " << (i + 1) << " / " << layerCount << endl;
        layers[i].CalculateChanges();
    }
}

void ConvolutionalNetwork::Improve()
{
    for(int i = 0; i < layerCount; i ++)
    {
        layers[i].Improve();
    }
}