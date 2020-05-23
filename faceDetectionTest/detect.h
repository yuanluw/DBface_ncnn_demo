#pragma once

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>
#include<vector>
#include<algorithm>
#include<string>
#include<net.h>
#include<platform.h>
#include "myutils.h"
using namespace cv;

class detect
{
public:
	detect(float thresh, float iou);
	vector<Obj> getObjs(Mat img);

private:
	ncnn::Net net;
	float THRESHOLD;
	float IOU;
	float STRIDE = 4;
	myutils util;
	ncnn::Mat norm(Mat img);
	void genIds(ncnn::Mat hm, ncnn::Mat hmPool, int w, double thresh, vector<Id>& ids);
	void decode(int w, vector<Id> ids, ncnn::Mat tlrb, ncnn::Mat landmark, vector<Obj>& objs);
};

