#include "ImageLoader.cuh"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

vector<string> splitString(string i_text, string i_delimiter)
{
    string text = i_text;

    vector<string> textList;

    size_t pos = 0;

    while ((pos = text.find(i_delimiter)) != string::npos) {
        string subText = text.substr(0, pos);
        textList.push_back(subText);
        text.erase(0, pos + i_delimiter.length());
    }
    textList.push_back(text);

    return textList;
}

string removeChracters(string i_text, vector<char> i_characters)
{
    string text;

    for(int i = 0; i < i_text.size(); i ++)
    {
        bool isInCharacterList = false;
        for(int j = 0; j < i_characters.size(); j ++)
        {
            if(i_text[i] == i_characters[j])
            {
                isInCharacterList = true;
            }
        }

        if(!isInCharacterList)
        {
            text += i_text[i];
        }
    }

    return text;
}

float clip(float value, float minValue, float maxValue)
{
    return min(max(value, minValue), maxValue);
}

BoundingBox::BoundingBox()
{
}

BoundingBox::BoundingBox(int i_minX, int i_maxX, int i_minY, int i_maxY)
{
    minX = i_minX;
    maxX = i_maxX;
    minY = i_minY;
    maxY = i_maxY;
}

int BoundingBox::getAreaOfIntersection(BoundingBox* i_boundingBox)
{
    int xOverlap = max(min(maxX, i_boundingBox->maxX) - max(minX, i_boundingBox->minX), 0);
    int yOverlap = max(min(maxY, i_boundingBox->maxY) - max(minY, i_boundingBox->minY), 0);

    return xOverlap * yOverlap;
}

int BoundingBox::getCombinedArea(BoundingBox* i_boundingBox, int i_areaOfOverlap)
{
    int area = (maxX - minX) * (maxY - minY);
    int boundingBoxArea = (i_boundingBox->maxX - i_boundingBox->minX) * (i_boundingBox->maxY - i_boundingBox->minY);

    return area + boundingBoxArea - i_areaOfOverlap;
}

int BoundingBox::getSpanArea(BoundingBox* i_boundingBox)
{
    int width = max(maxX, i_boundingBox->maxX) - min(minX, i_boundingBox->minX);
    int heigth = max(maxY, i_boundingBox->maxY) - min(minY, i_boundingBox->minY);   

    return width * heigth; 
}

float BoundingBox::getRatioOfIntersection(BoundingBox* i_boundingBox)
{
    int areaOfIntersection = getAreaOfIntersection(i_boundingBox);
    int combinedArea = getCombinedArea(i_boundingBox, areaOfIntersection);
    int spanArea = getSpanArea(i_boundingBox);

    float intersectionOverUnion = (float) areaOfIntersection / (float) combinedArea;
    float zeroCorrector = ((float) (spanArea - combinedArea)) / (float) spanArea;

    return intersectionOverUnion - zeroCorrector;
}

bool BoundingBox::isIn(int x, int y)
{
    if (x < minX)
        return false;
    if (x > maxX)
        return false;
    if (y < minY)
        return false;
    if (y > maxY)
        return false;

    return true;
}

bool BoundingBox::isOnBounds(int x, int y)
{
    if(x < minX)
        return false;
    if(x > maxX)
        return false;
    if(y < minY)
        return false;
    if(y > maxY)
        return false;

    if(x != minX && x != maxX && y != minY && y != maxY)
        return false;

    return true;
}

BoundingBoxStorage::BoundingBoxStorage()
{
}

void BoundingBoxStorage::getAllBoundingBoxes(int i_minSize, int i_maxSize, float i_maxRatio, int i_imageSize, int i_skip, int i_sizeSkip)
{
    for(int x = 0; x < i_imageSize; x += i_skip)
    {
        for(int y = 0; y < i_imageSize; y += i_skip)
        {
            for(int w = i_minSize; w < i_maxSize; w += i_sizeSkip)
            {
                if(x + w >= i_imageSize)
                {
                    break;
                }

                for(int h = i_minSize; h < i_maxSize; h += i_sizeSkip)
                {
                    if(y + h >= i_imageSize)
                    {
                        break;
                    }

                    float ratio = (float) max(w, h) / (float) min(w, h);

                    if(ratio > i_maxRatio)
                    {
                        continue;
                    }
                    allBoundingBoxes.push_back(BoundingBox(x, x + w, y, y + h));
                }
            }
        }
    }
}

TrafficSign::TrafficSign()
{
}

TrafficSign::TrafficSign(string i_label, int i_minX, int i_minY, int i_maxX, int i_maxY)
{
    label = i_label;
    minX = i_minX;
    minY = i_minY;
    maxX = i_maxX;
    maxY = i_maxY;
}

void TrafficSign::printTrafficSign()
{
    cout << "Traffic sign: (label: " << label << ", bounding box: [" << minX << " - " << maxX << ", " << minY << " - " << maxY << "])" << endl;
}

Annotation::Annotation()
{
}

void Annotation::loadAnnotation(string i_fileName)
{
    float width = 0;
    float height = 0;

    imageWidth = imageSize;
    imageHeight = imageSize;

    fstream newfile;

    newfile.open(i_fileName,ios::in);
    if (newfile.is_open())
    {
        string line;

        TrafficSign* trafficSign = NULL;

        while(getline(newfile, line))
        {
            line = removeChracters(line, {' ', ',', '"'});

            vector<string> words = splitString(line, ":");

            if(words[0] == "width")
            {
                width = stof(words[1]);
                continue;
            }
            
            if(words[0] == "height")
            {
                height = stof(words[1]);
                continue;
            }
            
            if(words[0] == "label")
            {
                trafficSigns.push_back(TrafficSign());
                trafficSign = &trafficSigns[trafficSigns.size() - 1];

                trafficSign->label = words[1];
                continue;
            }

            if (trafficSign == NULL)
            {
                continue;
            }

            if(words[0] == "xmin")
            {
                trafficSign->minX = (stof(words[1]));
                continue;
            }
            
            if(words[0] == "xmax")
            {
                trafficSign->maxX = (stof(words[1]));
                continue;
            }
            
            if(words[0] == "ymin")
            {
                trafficSign->minY = (stof(words[1]));
                continue;
            }
            
            if(words[0] == "ymax")
            {
                trafficSign->maxY = (stof(words[1]));

                if(trafficSign->maxY - trafficSign->minY < 8)
                {
                    trafficSigns.pop_back();
                }
                else if(trafficSign->maxX - trafficSign->minX < 8)
                {
                    trafficSigns.pop_back();
                }
                continue;
            }
        }
        newfile.close();
    }

    //cout << "loaded annotation" << endl;
}

void Annotation::setTransformedAnnotation(vector<TrafficSign>* i_trafficSigns, int i_x, int i_y, int i_imageWidth, int i_imageHeight)
{
    for (int i = 0; i < i_trafficSigns->size(); i++)
    {
        TrafficSign* addedSign = &trafficSigns[i];

        float factorX = i_imageWidth / imageWidth;
        float factorY = i_imageHeight / imageHeight;

        float newMinX = (addedSign->minX - i_x) * factorX;
        float newMinY = (addedSign->minY - i_y) * factorY;
        float newMaxX = (addedSign->maxX - i_x) * factorX;
        float newMaxY = (addedSign->maxY - i_y) * factorY;

        trafficSigns.push_back(TrafficSign(addedSign->label, newMinX, newMinY, newMaxX, newMaxY));
    }
}

vector<float>* Annotation::getBoundingBoxRating(BoundingBoxStorage* i_allBoundingBoxes)
{
    ratings = vector<float> (0);

    float delta = 0.1f;

    for(int i = 0; i < i_allBoundingBoxes->allBoundingBoxes.size(); i ++)
    {
        float bestRating = -1;

        for(int j = 0; j < trafficSigns.size(); j ++)
        {
            float rating = i_allBoundingBoxes->allBoundingBoxes[i].getRatioOfIntersection(&trafficSigns[j]);

            if(rating >= trafficSigns[j].maxOverlap - delta)
            {
                bestRating = 1;
                break;
            }

            if(rating >= bestRating)
            {
                bestRating = rating;
            }
        }

        if(bestRating > 0.5f)
        {
            ratings.push_back(1);
        }
        else if(bestRating <= 0.0f && rand() % 100 > 50)
        {
            ratings.push_back(0);
        }
        else
        {
            ratings.push_back(-1); 
        }

    }

    return &ratings;
}

vector<float>* Annotation::getPixelRating(int i_inputSize, int i_outputSize)
{
    ratings = vector<float>(0);

    for (int y = 0; y < i_outputSize; y++)
    {
        for (int x = 0; x < i_outputSize; x++)
        {
            float rX = (float) x / (float) i_outputSize;
            float rY = (float) y / (float) i_outputSize;

            int nX = rX * i_inputSize;
            int nY = rY * i_inputSize;

            bool isIn = false;

            for (int i = 0; i < trafficSigns.size(); i++)
            {
                TrafficSign* trafficSign = &trafficSigns[i];
                if (trafficSign->isIn(nX, nY))
                {
                    isIn = true;
                    break;
                }
            }

            ratings.push_back(isIn ? 1 : 0);
        }
    }

    return &ratings;
}

void Annotation::printAnnotation()
{
    cout << "Annotation" << endl;
    for(int i = 0; i < trafficSigns.size(); i ++)
    {
        trafficSigns[i].printTrafficSign();
    }
}

Image::Image()
{

}

Image::Image(string i_path, int i_imageSize)
{
    imageSize = i_imageSize;

    string path = removeChracters(i_path, { '"' });
    //cout << "Path: " << path << endl;

    vector<string> filePaths = splitString(path, "\\");

    for (int i = 0; i < filePaths.size() - 2; i++)
    {
        sourcePath += filePaths[i] + "\\";
    }

    fileName = filePaths[filePaths.size() - 1].substr(0, filePaths[filePaths.size() - 1].size() - 4);

    //cout << "Source: " << sourcePath << ", Image name: " << fileName << endl;

    string imagePath = sourcePath + "images\\" + fileName + ".jpg";

    //cout << "Image path: " << imagePath << endl;

    //cout << "Annotation path: " << sourcePath << "annotations\\" << fileName << ".json" << endl;

    loadAnnotation(sourcePath + "annotations\\" + fileName + ".json");

    if(trafficSigns.size() == 0)
    {
        cout << "No traffic signs found" << endl;
        return;
    }

    loadImage(imagePath);
    resizeImage(i_imageSize);
}

void Image::loadImage(string i_path)
{
    const char* pathChars = i_path.c_str();

    int bpp;

    uint8_t* rgb_image = stbi_load(pathChars, &imageWidth, &imageHeight, &bpp, 3);

    for(int y = 0; y < imageHeight; y ++)
    {
        for(int x = 0; x < imageWidth; x ++)
        {
            unsigned bytePerPixel = 3;
            unsigned char* pixelData = rgb_image + (x + imageWidth * y) * bytePerPixel;

            rChannel.push_back((float)pixelData[0] / 255.0f);
            gChannel.push_back((float)pixelData[1] / 255.0f);
            bChannel.push_back((float)pixelData[2] / 255.0f);
        }
    }

    stbi_image_free(rgb_image);
}

void Image::resizeImage(int i_imageSize)
{
    vector<float> rawRChannel = rChannel;
    vector<float> rawGChannel = gChannel;
    vector<float> rawBChannel = bChannel;

    float ratioX = (float)imageWidth / (float)i_imageSize;

    int gaussianKernelWidth = max(ratioX / 2, 1.0f);
    int gaussianKernelSizeX = gaussianKernelWidth * 2 + 1;
    float deviationX = max(((float)gaussianKernelWidth / 3.0f), 1.0f);
    float sqrDeviationX = deviationX * deviationX;

    vector<float> gaussianKernelX = vector<float>(gaussianKernelSizeX);

    float sum = 0;
    
    for (int dX = -gaussianKernelWidth; dX < gaussianKernelWidth + 1; dX++)
    {
        float sqrDistance = dX * dX;
        float gaussian = exp(-sqrDistance / (2.0f * sqrDeviationX)) / sqrt(2.0f * 3.142f * sqrDeviationX);
        sum += gaussian;

        int kernelIndex = dX + gaussianKernelWidth;
        gaussianKernelX[kernelIndex] = gaussian;
    }

    float ratioY = (float)imageHeight / (float)i_imageSize;

    int gaussianKernelHeight = max(ratioY / 2, 1.0f);
    int gaussianKernelSizeY = gaussianKernelHeight * 2 + 1;
    float deviationY = max(((float)gaussianKernelHeight / 3.0f), 1.0f);
    float sqrDeviationY = deviationY * deviationY;

    vector<float> gaussianKernelY = vector<float>(gaussianKernelSizeY);

    for (int dY = -gaussianKernelHeight; dY < gaussianKernelHeight + 1; dY++)
    {
        float sqrDistance = dY * dY;
        float gaussian = exp(-sqrDistance / (2.0f * sqrDeviationY)) / sqrt(2.0f * 3.142f * sqrDeviationY);
        sum += gaussian;

        int kernelIndex = dY + gaussianKernelHeight;
        gaussianKernelY[kernelIndex] = gaussian;
    }

    rChannel = vector<float> (i_imageSize * imageHeight);
    gChannel = vector<float> (i_imageSize * imageHeight);
    bChannel = vector<float> (i_imageSize * imageHeight);

    for(int y = 0; y < imageHeight; y ++)
    {
        for(int x = 0; x < i_imageSize; x ++)
        {
            int pixelIndex = x + i_imageSize * y;

            float rawX = ratioX * x;
            float rawY = y;

            for (int dX = -gaussianKernelWidth; dX < gaussianKernelWidth + 1; dX++)
            {
                int convolutedX = rawX + dX;

                if (convolutedX < 0 || convolutedX >= imageWidth)
                {
                    continue;
                }

                int convolutedIndex = convolutedX + imageWidth * rawY;

                float factor = gaussianKernelX[dX + gaussianKernelWidth];
                    
                rChannel[pixelIndex] += rawRChannel[convolutedIndex] * factor;
                gChannel[pixelIndex] += rawGChannel[convolutedIndex] * factor;
                bChannel[pixelIndex] += rawBChannel[convolutedIndex] * factor;
            }
        }
    }

    rawRChannel = rChannel;
    rawGChannel = gChannel;
    rawBChannel = bChannel;

    rChannel = vector<float>(i_imageSize * i_imageSize);
    gChannel = vector<float>(i_imageSize * i_imageSize);
    bChannel = vector<float>(i_imageSize * i_imageSize);

    for (int y = 0; y < i_imageSize; y++)
    {
        for (int x = 0; x < i_imageSize; x++)
        {
            int pixelIndex = x + i_imageSize * y;

            float rawX = x;
            float rawY = ratioY * y;

            for (int dY = -gaussianKernelHeight; dY < gaussianKernelHeight + 1; dY++)
            {
                int convolutedY = rawY + dY;

                if (convolutedY < 0 || convolutedY >= imageHeight)
                {
                    continue;
                }

                int convolutedIndex = rawX + i_imageSize * convolutedY;

                float factor = gaussianKernelY[dY + gaussianKernelHeight];

                rChannel[pixelIndex] += rawRChannel[convolutedIndex] * factor;
                gChannel[pixelIndex] += rawGChannel[convolutedIndex] * factor;
                bChannel[pixelIndex] += rawBChannel[convolutedIndex] * factor;
            }
        }
    }

    for (int i = 0; i < trafficSigns.size(); i++)
    {
        TrafficSign* addedSign = &trafficSigns[i];

        float factorX = (float)i_imageSize / (float)imageWidth;
        float factorY = (float)i_imageSize / (float)imageHeight;

        float newMinX = clip((addedSign->minX) * factorX, 0, i_imageSize);
        float newMinY = clip((addedSign->minY) * factorY, 0, i_imageSize);
        float newMaxX = clip((addedSign->maxX) * factorX, 0, i_imageSize);
        float newMaxY = clip((addedSign->maxY) * factorY, 0, i_imageSize);

        addedSign->minX = newMinX;
        addedSign->minY = newMinY;
        addedSign->maxX = newMaxX;
        addedSign->maxY = newMaxY;
    }

    imageWidth = i_imageSize;
    imageHeight = i_imageSize;

    imageSize = i_imageSize;
}

Image Image::getSubImage(int i_x, int i_y, int i_imageWidth, int i_imageHeight)
{
    Image outputImage;

    if (i_x < 0 || i_x + i_imageWidth >= imageWidth || i_y < 0 || i_y + i_imageHeight >= imageWidth)
    {
        cout << "Error reading coordinates" << endl;
        cout << i_x << ", " << i_y << ", " << i_imageWidth << " / " << imageWidth << ", " << i_imageHeight << " / " << imageHeight << endl;

        return outputImage;
    }

    outputImage.rChannel = vector<float>(i_imageWidth * i_imageHeight);
    outputImage.gChannel = vector<float>(i_imageWidth * i_imageHeight);
    outputImage.bChannel = vector<float>(i_imageWidth * i_imageHeight);

    for (int y = 0; y < i_imageHeight; y++)
    {
        for (int x = 0; x < i_imageWidth; x++)
        {
            int pixelIndex = (x + i_x) + imageWidth * (y + i_y);
            int outputIndex = x + i_imageWidth * y;

            outputImage.rChannel[outputIndex] = rChannel[pixelIndex];
            outputImage.gChannel[outputIndex] = gChannel[pixelIndex];
            outputImage.bChannel[outputIndex] = bChannel[pixelIndex];
        }
    }

    for (int i = 0; i < trafficSigns.size(); i++)
    {
        TrafficSign* addedSign = &trafficSigns[i];

        float factorX = imageWidth / i_imageWidth;
        float factorY = imageHeight / i_imageHeight;

        float newMinX = clip((addedSign->minX - i_x), 0, i_imageWidth);
        float newMinY = clip((addedSign->minY - i_y), 0, i_imageHeight);
        float newMaxX = clip((addedSign->maxX - i_x), 0, i_imageWidth);
        float newMaxY = clip((addedSign->maxY - i_y), 0, i_imageHeight);

        outputImage.trafficSigns.push_back(TrafficSign(addedSign->label, newMinX, newMinY, newMaxX, newMaxY));
    }

    outputImage.sourcePath = sourcePath;
    outputImage.fileName = fileName;

    outputImage.imageWidth = i_imageWidth;
    outputImage.imageHeight = i_imageHeight;

    outputImage.imageSize = i_imageWidth;

    int pixelOverlap = 0;

    for (int y = 0; y < outputImage.imageHeight; y++)
    {
        for (int x = 0; x < outputImage.imageWidth; x++)
        {
            int pixelIndex = x + outputImage.imageWidth * y;

            for (int i = 0; i < outputImage.trafficSigns.size(); i++)
            {
                if (outputImage.trafficSigns[i].isIn(x, y))
                {
                    pixelOverlap++;
                    break;
                }
            }
        }
    }

    outputImage.objectPercentage = (float)pixelOverlap / (float)(outputImage.imageWidth * outputImage.imageHeight);

    return outputImage;
}

vector<Image> Image::getSubImages(vector<int> i_sizes, vector<float> i_ratios, int i_skipping)
{
    vector<Image> subImages;
    for (int y = 0; y < imageHeight; y += i_skipping)
    {
        for (int x = 0; x < imageWidth; x += i_skipping)
        {
            for (int i = 0; i < i_sizes.size(); i++)
            {
                for (int j = 0; j < i_ratios.size(); j++)
                {
                    int maxX = x + i_sizes[i];
                    int maxY = y + i_sizes[i] * i_ratios[j];

                    if (maxX >= imageWidth || maxY >= imageHeight)
                        continue;

                    subImages.push_back(getSubImage(x, y, i_sizes[i], i_sizes[i] * i_ratios[j]));
                }
            }
        }
    }

    return subImages;
}

vector<vector<float>*> Image::getImageData()
{
    return {&rChannel, &gChannel, &bChannel};
}

void Image::setOptimalResults(BoundingBoxStorage* i_allBoundingBoxes)
{

    for(int j = 0; j < trafficSigns.size(); j ++)
    {
        float bestRating = -1;
        int bestIndex = 0;

        for(int i = 0; i < i_allBoundingBoxes->allBoundingBoxes.size(); i ++)
        {
            float rating = i_allBoundingBoxes->allBoundingBoxes[i].getRatioOfIntersection(&trafficSigns[j]);

            if(rating > bestRating)
            {
                bestRating = rating;
                bestIndex = i;
            }
        }

        optimalResults.push_back(&i_allBoundingBoxes->allBoundingBoxes[bestIndex]);

        trafficSigns[j].maxOverlap = bestRating;

        //cout << "Optimal bounding box " << bestIndex << " iou: " << bestRating << endl;
    }
}

void Image::setBestNetResults(vector<float>* i_results)
{
    float bestResult = -10;
    int bestIndex = 0;

    for(int i = 0; i < i_results->size(); i ++)
    {
        if ((*i_results)[i] > bestResult)
        {
            bestIndex = i;
            bestResult = (*i_results)[i];
        }

        if((*i_results)[i] > 0.7)
        {
            bestNetResults.push_back(i);
        }
    }

    cout << "Best index " << bestIndex << ", confidence: " << bestResult << endl;

    for (int y = 0; y < imageSize; y++)
    {
        for (int x = 0; x < imageSize; x++)
        {
            int oY = (float)y / (float)imageSize * 61;
            int oX = (float)x / (float)imageSize * 61;

            int oIndex = oX + 61 * oY;

            pixelOutput.push_back((*i_results)[oIndex]);
        }
    }

    cout << "Pixel area " << pixelOutput.size() << endl;

}


void Image::printImage()
{
    //cout << "Image: (width:" << imageWidth << ", height: " << imageHeight << ")" << endl;

    /*for(int y = 0; y < imageSize; y ++)
    {
        for(int x = 0; x < imageSize; x ++)
        {
            int pixelIndex = x + imageSize * y;

            float r = rChannel[pixelIndex];
            float g = gChannel[pixelIndex];
            float b = bChannel[pixelIndex];

            int outputIndex = (int) (10.0f * (max(r, max(g, b)) + min( r, min(g, b))) / 2.0f);

            vector<char> outputCharacters = {' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'};

            cout << outputCharacters[outputIndex];

        }
        cout << endl;
    }*/

    uint8_t* pixels = new uint8_t[imageSize * imageSize * 3];

    int index = 0;

    for(int y = 0; y < imageSize; y ++)
    {
        for(int x = 0; x < imageSize; x ++)
        {
            int pixelIndex = x + imageSize * y;

            pixels[index ++] = rChannel[pixelIndex] * 255.0f;
            pixels[index ++] = gChannel[pixelIndex] * 255.0f;
            pixels[index ++] = bChannel[pixelIndex] * 255.0f;

            for(int i = 0; i < trafficSigns.size(); i ++)
            {
                if(trafficSigns[i].isOnBounds(x, y))
                {
                    pixels[index - 1] = 0;
                    pixels[index - 2] = 0;
                    pixels[index - 3] = 255.0f;
                }
            }
        }
    }


    for (int i = 0; i < pixelOutput.size(); i++)
    {
        float darknessFactor = max(pixelOutput[i], 0.1f);
        
        //pixels[i * 3 + 0] = (float)pixels[i * 3 + 0] * darknessFactor;
        //pixels[i * 3 + 1] = (float)pixels[i * 3 + 1] * darknessFactor;
        //pixels[i * 3 + 2] = (float)pixels[i * 3 + 2] * darknessFactor;
    }

    string imagePath = sourcePath + "results\\" + fileName + ".jpg";

    const char* pathChars = imagePath.c_str();

    stbi_write_jpg(pathChars, imageSize, imageSize, 3, pixels, 100);
}