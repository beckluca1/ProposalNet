#include "main.cuh"

string removeChracter(string i_text, char i_character)
{
    string text;

    for(int i = 0; i < i_text.size(); i ++)
    {
        if(i_text[i] != i_character)
        {
            text += i_text[i];
        }
    }

    return text;
}

int main()
{
    srand (static_cast <unsigned> (time(0)));

    BoundingBoxStorage allBoundingBoxes;
    allBoundingBoxes.getAllBoundingBoxes(32, 96, 2.0f, 512, 16, 32);

    int boundingBoxCount = allBoundingBoxes.allBoundingBoxes.size();

    cout << "Welcome to PRN" << endl;

    cout << "Found " << boundingBoxCount << " bounding boxes" << endl;

    vector<MapType> layerType = {Input, Convolutional, Pooling, Convolutional, Pooling, Convolutional, Pooling, Connected, Connected};
    vector<int> mapCount = {3, 8, 8, 16, 16, 32, 32, 1, 1};
    vector<int> reduction = {64, 3, 2, 4, 2, 3, 2, 100, 2};

    ConvolutionalNetwork net = ConvolutionalNetwork(layerType, mapCount, reduction);

    cout << "Enter path for image" << endl;

    string path;
    getline(cin, path);

    path = removeChracter(path, '"');

    DIR *dir;
    struct dirent *diread;
    vector<string> fileNames;

    if ((dir = opendir(path.c_str())) != nullptr) 
    {
        while ((diread = readdir(dir)) != nullptr) 
        {
            fileNames.push_back(string(diread->d_name));
        }
        closedir (dir);
    } 
    else 
    {
        perror ("opendir");
        return EXIT_FAILURE;
    }

    for(int i = 0; i < fileNames.size(); i ++)
    {
        string fileString = fileNames[i];

        if(fileString[0] == '.')
            continue;

        string imagePath = path + "\\" + fileString;
        
        Image image = Image(imagePath, 1024);

        if(image.trafficSigns.size() == 0)
            continue;

        vector<Frame> subFrames = image.getSubImageFrames({ 32,48,64,96,128 }, { 1.0 }, 8);

        int bestIndex = 0;
        float bestObjectPercentage = 0;

        vector<Frame> positiveSamples = vector<Frame> ();
        vector<Frame> negativeSamples = vector<Frame>();
        for (int j = 0; j < subFrames.size(); j++)
        {
            if (subFrames[j].percentage > bestObjectPercentage)
            {
                bestIndex = j;
                bestObjectPercentage = subFrames[j].percentage;
            }

            if (subFrames[j].percentage > 0.7)
            {
                positiveSamples.push_back(subFrames[j]);

            }
            else if(subFrames[j].percentage == 0)
            {
                negativeSamples.push_back(subFrames[j]);
            }
        }

        int sampleSize = min(positiveSamples.size(), negativeSamples.size());

        if (sampleSize <= 0)
        {
            continue;
        }

        image.loadImage(1024);

        float correct = 0;

        for (int j = 0; j < sampleSize; j++)
        {             
            int positiveIndex = rand() % positiveSamples.size();
            Image positiveSample = image.getSubImage(positiveSamples[positiveIndex]);
            positiveSample.resizeImage(64);

            //positiveSample.printImage(j);

            vector<float> positiveObjectPercentage = { 1, 0 };
            net.Train(positiveSample.getImageData(), &positiveObjectPercentage);
            
            //cout << (*net.GetValues(net.layerCount - 1)[0])[0] << ", " << (*net.GetValues(net.layerCount - 1)[0])[1] << endl;

            vector<vector<float>*> inputNet = net.layers[0].GetValues();

            correct += pow(1 - (*net.GetValues(net.layerCount - 1)[0])[0],2) + pow((*net.GetValues(net.layerCount - 1)[0])[1],2);

            positiveSamples.erase(positiveSamples.begin() + positiveIndex);


            int negativeIndex = rand() % negativeSamples.size();
            Image negativeSample = image.getSubImage(negativeSamples[negativeIndex]);
            negativeSample.resizeImage(64);

            vector<float> negativeObjectPercentage = { 0, 1 };
            net.Train(negativeSample.getImageData(), &negativeObjectPercentage);

            //cout << (*net.GetValues(net.layerCount - 1)[0])[0] << ", " << (*net.GetValues(net.layerCount - 1)[0])[1] << endl;

            correct += pow(1 - (*net.GetValues(net.layerCount - 1)[0])[1],2) + pow((*net.GetValues(net.layerCount - 1)[0])[0],2);


            negativeSamples.erase(negativeSamples.begin() + negativeIndex);
        }

        net.Improve();


        if (sampleSize > 0)
        {
            float rating = (float)correct / (float)(2 * sampleSize);
            cout << "Rating: " << rating << endl;

        }

        //image.resizeImage(128);
        //image.printImage(-1);

        image.setOptimalResults(&allBoundingBoxes);
            
        vector<float>* netOutput = net.GetValues(net.layerCount - 1)[0];
    }

    return 0;
}