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
    vector<int> reduction = {128, 3, 2, 4, 2, 3, 2, 100, 2};

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

        cout << "File: " << fileString << endl;

        string imagePath = path + "\\" + fileString;
        //cout << imagePath << endl;

        Image image = Image(imagePath, 1024);

        if(image.trafficSigns.size() == 0)
            continue;

        vector<Image> subImages = image.getSubImages({ 64,128,256 }, { 1.0 }, 64);

        int bestIndex = 0;
        float bestObjectPercentage = 0;

        vector<Image> positiveSamples = vector<Image> ();
        vector<Image> negativeSamples = vector<Image>();
        for (int j = 0; j < subImages.size(); j++)
        {
            if (subImages[j].objectPercentage > bestObjectPercentage)
            {
                bestIndex = j;
                bestObjectPercentage = subImages[j].objectPercentage;
            }

            if (subImages[j].objectPercentage > 0.3)
            {
                positiveSamples.push_back(subImages[j]);

            }
            else if(subImages[j].objectPercentage == 0)
            {
                negativeSamples.push_back(subImages[j]);
            }
        }

        image = subImages[bestIndex];

        int sampleSize = min(positiveSamples.size(), negativeSamples.size());

        float correct = 0;

        for (int j = 0; j < sampleSize; j++)
        {
            int positiveIndex = rand() % positiveSamples.size();

            positiveSamples[positiveIndex].resizeImage(128);
            vector<float> positiveObjectPercentage = { 1, 0 };// positiveSamples[positiveIndex]->objectPercentage};
            net.Train(positiveSamples[positiveIndex].getImageData(), &positiveObjectPercentage);

            vector<vector<float>*> inputNet = net.layers[0].GetValues();
            //cout << "Positive test: " << (*net.GetValues(net.layerCount - 1)[0])[0] << ", " << (*net.GetValues(net.layerCount - 1)[0])[1] << " / " << positiveSamples[positiveIndex].objectPercentage << endl;

            correct += 1 - (*net.GetValues(net.layerCount - 1)[0])[0] + (*net.GetValues(net.layerCount - 1)[0])[1];

            positiveSamples.erase(positiveSamples.begin() + positiveIndex);


            int negativeIndex = rand() % negativeSamples.size();

            negativeSamples[negativeIndex].resizeImage(128);
            vector<float> negativeObjectPercentage = { 0, 1 };
            net.Train(negativeSamples[negativeIndex].getImageData(), &negativeObjectPercentage);

            //cout << "Negative test: " << (*net.GetValues(net.layerCount - 1)[0])[0] << ", " << (*net.GetValues(net.layerCount - 1)[0])[1] << " / " << negativeSamples[negativeIndex].objectPercentage << endl;

            correct += 1 - (*net.GetValues(net.layerCount - 1)[0])[1] + (*net.GetValues(net.layerCount - 1)[0])[0];


            negativeSamples.erase(negativeSamples.begin() + negativeIndex);
        }

        net.Improve();


        if (sampleSize > 0)
        {
            float rating = (float)correct / (float)(2 * sampleSize);
            cout << "Rating: " << rating << endl;

        }

        image.resizeImage(128);

        //cout << "Get image data" << endl;


        //cout << "Rate bounding boxes" << endl;

        image.setOptimalResults(&allBoundingBoxes);
            
        //cout << "Train net" << endl;

        vector<float>* netOutput = net.GetValues(net.layerCount - 1)[0];
        //image.setBestNetResults(netOutput);

        //cout << "Improve net" << endl;

        if(image.objectPercentage > 0.3)
        {
            image.printImage();
        }
    }

    return 0;
}