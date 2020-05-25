#pragma once

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>
#include<vector>
#include<algorithm>
#include<string>
#include<net.h>
#include<platform.h>

using namespace std;
using namespace cv;

class arcFace
{
public:
	arcFace();
	vector<float> getFeature(Mat img);
private:
	ncnn::Net net;
	ncnn::Mat norm(Mat img);
};

