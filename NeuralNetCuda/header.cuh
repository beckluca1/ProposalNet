#ifndef HEADER_INCLUDED
#define HEADER_INCLUDED

#include <iostream>
#include <iomanip>
using namespace std;

#include <math.h>
#include <vector>
#include <ctime>
#include <algorithm>
#include <stdint.h>
#include <fstream>
#include <string>

#include "dirent.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "NeuralNet/NeuralNet.cuh"
#include "DataLoader/ImageLoader.cuh"

#include "kernel.cu"

#endif