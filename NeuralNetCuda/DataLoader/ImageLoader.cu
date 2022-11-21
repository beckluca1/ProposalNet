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
                annotationWidth = stof(words[1]);
                continue;
            }
            
            if(words[0] == "height")
            {
                annotationHeight = stof(words[1]);
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

Frame::Frame()
{

}

Frame::Frame(int i_x, int i_y, int i_imageWidth, int i_imageHeight)
{
    x = i_x;
    y = i_y;
    imageWidth = i_imageWidth;
    imageHeight = i_imageHeight;
}

Image::Image()
{
    imageSize = 0;
}

Image::Image(string i_path, int i_imageSize)
{
    setImageSize(i_imageSize);

    string path = removeChracters(i_path, { '"' });

    vector<string> filePaths = splitString(path, "\\");

    for (int i = 0; i < filePaths.size() - 2; i++)
    {
        sourcePath += filePaths[i] + "\\";
    }

    fileName = filePaths[filePaths.size() - 1].substr(0, filePaths[filePaths.size() - 1].size() - 4);

    imagePath = sourcePath + "images\\" + fileName + ".jpg";

    loadAnnotation(sourcePath + "annotations\\" + fileName + ".json");
    resizeAnnotation(i_imageSize);

    if(trafficSigns.size() == 0)
    {
        return;
    }
}

void Image::setImageSize(int i_imageSize)
{
    imageWidth = i_imageSize;
    imageHeight = i_imageSize;

    imageSize = i_imageSize;
}

void Image::loadImage(int i_imageSize)
{
    const char* pathChars = imagePath.c_str();

    int bpp;

    int rawWidth;
    int rawHeight;

    uint8_t* rgb_image = stbi_load(pathChars, &rawWidth, &rawHeight, &bpp, 3);

    float factorX = (float)rawWidth / (float)i_imageSize;
    float factorY = (float)rawHeight / (float)i_imageSize;

    for(int y = 0; y < i_imageSize; y ++)
    {
        for(int x = 0; x < i_imageSize; x ++)
        {
            unsigned bytePerPixel = 3;
            unsigned char* pixelData = rgb_image + (int(x * factorX) + rawWidth * int(y * factorY)) * bytePerPixel;

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

    rChannel = vector<float>(i_imageSize * i_imageSize);
    gChannel = vector<float>(i_imageSize * i_imageSize);
    bChannel = vector<float>(i_imageSize * i_imageSize);

    float factorX = (float)imageWidth / (float)i_imageSize;
    float factorY = (float)imageHeight / (float)i_imageSize;

    for (int y = 0; y < i_imageSize; y++)
    {
        for (int x = 0; x < i_imageSize; x++)
        {
            int newIndex = x + i_imageSize * y;
            int pixelIndex = int(x * factorX) + imageWidth * int(y * factorY);

            rChannel[newIndex] = rawRChannel[pixelIndex];
            gChannel[newIndex] = rawGChannel[pixelIndex];
            bChannel[newIndex] = rawBChannel[pixelIndex];
        }
    }

    resizeAnnotation(i_imageSize);
    setImageSize(i_imageSize);
}

void Image::resizeAnnotation(int i_imageSize)
{
    for (int i = 0; i < trafficSigns.size(); i++)
    {
        TrafficSign* addedSign = &trafficSigns[i];

        float factorX = (float)i_imageSize / (float)annotationWidth;
        float factorY = (float)i_imageSize / (float)annotationHeight;

        float newMinX = clip((addedSign->minX) * factorX, 0, i_imageSize);
        float newMinY = clip((addedSign->minY) * factorY, 0, i_imageSize);
        float newMaxX = clip((addedSign->maxX) * factorX, 0, i_imageSize);
        float newMaxY = clip((addedSign->maxY) * factorY, 0, i_imageSize);

        addedSign->minX = newMinX;
        addedSign->minY = newMinY;
        addedSign->maxX = newMaxX;
        addedSign->maxY = newMaxY;
    }

    annotationWidth = i_imageSize;
    annotationHeight = i_imageSize;
}

Image Image::getSubImage(Frame i_frame)
{
    Image outputImage;

    if (i_frame.x < 0 || i_frame.x + i_frame.imageWidth >= imageWidth || i_frame.y < 0 || i_frame.y + i_frame.imageHeight >= imageHeight)
    {
        cout << "Error reading coordinates" << endl;
        cout << i_frame.x << ", " << i_frame.y << ", " << i_frame.imageWidth << " / " << imageWidth << ", " << i_frame.imageHeight << " / " << imageHeight << endl;

        return outputImage;
    }

    outputImage.rChannel = vector<float>(i_frame.imageWidth * i_frame.imageHeight);
    outputImage.gChannel = vector<float>(i_frame.imageWidth * i_frame.imageHeight);
    outputImage.bChannel = vector<float>(i_frame.imageWidth * i_frame.imageHeight);

    for (int y = 0; y < i_frame.imageHeight; y++)
    {
        for (int x = 0; x < i_frame.imageWidth; x++)
        {
            int pixelIndex = (x + i_frame.x) + imageWidth * (y + i_frame.y);
            int outputIndex = x + i_frame.imageWidth * y;

            outputImage.rChannel[outputIndex] = rChannel[pixelIndex];
            outputImage.gChannel[outputIndex] = gChannel[pixelIndex];
            outputImage.bChannel[outputIndex] = bChannel[pixelIndex];
        }
    }

    for (int i = 0; i < trafficSigns.size(); i++)
    {
        TrafficSign* addedSign = &trafficSigns[i];

        float factorX = imageWidth / i_frame.imageWidth;
        float factorY = imageHeight / i_frame.imageHeight;

        float newMinX = clip((addedSign->minX - i_frame.x), 0, i_frame.imageWidth);
        float newMinY = clip((addedSign->minY - i_frame.y), 0, i_frame.imageHeight);
        float newMaxX = clip((addedSign->maxX - i_frame.x), 0, i_frame.imageWidth);
        float newMaxY = clip((addedSign->maxY - i_frame.y), 0, i_frame.imageHeight);

        outputImage.trafficSigns.push_back(TrafficSign(addedSign->label, newMinX, newMinY, newMaxX, newMaxY));
    }

    outputImage.sourcePath = sourcePath;
    outputImage.fileName = fileName;

    outputImage.imageWidth = i_frame.imageWidth;
    outputImage.imageHeight = i_frame.imageHeight;

    outputImage.imageSize = i_frame.imageWidth;

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

Frame Image::getSubImageFrame(int i_x, int i_y, int i_imageWidth, int i_imageHeight)
{
    Frame outputFrame = Frame(i_x, i_y, i_imageWidth, i_imageHeight);

    if (i_x < 0 || i_x + i_imageWidth >= imageWidth || i_y < 0 || i_y + i_imageHeight >= imageWidth)
    {
        cout << "Error reading coordinates" << endl;
        cout << i_x << ", " << i_y << ", " << i_imageWidth << " / " << imageWidth << ", " << i_imageHeight << " / " << imageHeight << endl;

        return outputFrame;
    }

    float maxOverlap = 0;

    vector<TrafficSign> signs;

    for (int i = 0; i < trafficSigns.size(); i++)
    {
        TrafficSign* addedSign = &trafficSigns[i];

        int unionX = max(addedSign->maxX, i_x + i_imageWidth) - min(i_x, addedSign->minX);
        int unionY = max(addedSign->maxY, i_y + i_imageHeight) - min(i_y, addedSign->minY);
        int unionArea = unionX * unionY;

        int newMinX = clip((addedSign->minX - i_x), 0, i_imageWidth);
        int newMinY = clip((addedSign->minY - i_y), 0, i_imageHeight);
        int newMaxX = clip((addedSign->maxX - i_x), 0, i_imageWidth);
        int newMaxY = clip((addedSign->maxY - i_y), 0, i_imageHeight);

        int overlapX = min(addedSign->maxX, i_x + i_imageWidth) - max(addedSign->minX, i_x);
        int overlapY = min(addedSign->maxY, i_y + i_imageHeight) - max(addedSign->minY, i_y);

        if (overlapX <= 0 || overlapY <= 0)
        {
            continue;
        }

        float overlap = (float)(overlapX * overlapY) / (float)unionArea;

        if (overlap <= maxOverlap)
        {
            continue;
        }

        maxOverlap = overlap;
    }

    outputFrame.percentage = maxOverlap;
    
    return outputFrame;
}

vector<Image> Image::getSubImages(vector<Frame> i_frames)
{
    vector<Image> subImages;
    for (int i = 0; i < i_frames.size(); i++)
    {           
        subImages.push_back(getSubImage(i_frames[i]));
    }

    return subImages;
}

vector<Frame> Image::getSubImageFrames(vector<int> i_sizes, vector<float> i_ratios, int i_skipping)
{
    vector<Frame> subFrames;
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

                    subFrames.push_back(getSubImageFrame(x, y, i_sizes[i], i_sizes[i] * i_ratios[j]));
                }
            }
        }
    }

    return subFrames;
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

}


void Image::printImage(int i_index)
{
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

    string outputPath = sourcePath + "results\\" + fileName + to_string(i_index) + ".jpg";

    const char* pathChars = outputPath.c_str();

    stbi_write_jpg(pathChars, imageSize, imageSize, 3, pixels, 100);
}