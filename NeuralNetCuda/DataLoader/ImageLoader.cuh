#ifndef IMAGE_LOADER_INCLUDED
#define IMAGE_LOADER_INCLUDED

#include "../header.cuh"

class BoundingBox
{
    public:
        int minX;
        int maxX;
        int minY;
        int maxY;

        float maxOverlap;

        BoundingBox();
        BoundingBox(int i_minX, int i_maxX, int i_minY, int i_maxY);

        int getAreaOfIntersection(BoundingBox* i_boundingBox);
        int getCombinedArea(BoundingBox* i_boundingBox, int i_areaOfOverlap);
        int getSpanArea(BoundingBox* i_boundingBox);

        float getRatioOfIntersection(BoundingBox* i_boundingBox);

        bool isIn(int x, int y);
        bool isOnBounds(int x, int y);
};

class BoundingBoxStorage
{
    public:
        vector<BoundingBox> allBoundingBoxes;

        BoundingBoxStorage();

        void getAllBoundingBoxes(int i_minSize, int i_maxSize, float i_maxRatio, int i_imageSize, int i_skip, int i_sizeSkip);
};

class TrafficSign : public BoundingBox
{
    public:
        string label;

        TrafficSign();
        TrafficSign(string i_label, int i_minX, int i_minY, int i_maxX, int i_maxY);

        void printTrafficSign();
};

class Annotation
{
    public:
        int imageSize;

        int imageWidth;
        int imageHeight;

        float objectPercentage;

        vector<TrafficSign> trafficSigns;
        vector<float> ratings;

        Annotation();

        void loadAnnotation(string i_fileName);
        
        void setTransformedAnnotation(vector<TrafficSign>* i_trafficSigns, int i_x, int i_y, int i_imageWidth, int i_imageHeight);

        vector<float>* getBoundingBoxRating(BoundingBoxStorage* i_allBoundingBoxes);
        vector<float>* getPixelRating(int i_inputSize, int i_outputSize);

        void printAnnotation();
};

class Image : public Annotation
{
    public:
        string sourcePath;
        string fileName;

        vector<float> rChannel;
        vector<float> gChannel;
        vector<float> bChannel;

        vector<BoundingBox*> optimalResults;
        vector<int> bestNetResults;

        vector<float> pixelOutput;

        Image();
        Image(string i_path, int i_imageSize);

        void loadImage(string i_path);

        void resizeImage(int i_imageSize);

        Image getSubImage(int i_x, int i_y, int i_imageWidth, int i_imageHeight);
        vector<Image> getSubImages(vector<int> i_sizes, vector<float> i_ratios, int i_skipping);

        vector<vector<float>*> getImageData();

        void setOptimalResults(BoundingBoxStorage* i_allBoundingBoxes);
        void setBestNetResults(vector<float>* i_results);

        void printImage();
};

#endif