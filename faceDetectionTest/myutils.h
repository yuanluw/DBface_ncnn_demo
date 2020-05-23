#pragma once

#include<vector>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>
#include<algorithm>

using namespace std;
using namespace cv;

struct Box {
	float x, y, r, b;
};

struct Landmark {
	vector<float> x;
	vector<float> y;
};

struct Obj {
	double score;
	Box box;
	Landmark landmark;
};

struct Id {
	double score;
	int idx;
	int idy;
};

class myutils
{
public:
	vector<Obj> nms(vector<Obj> objs, float iou = 0.5);
	Mat pad(Mat img, int stride = 32);
	float getIou(Box a, Box b);
	float myExp(float v);
	void drawBbox(Mat& img, Obj o, Scalar textColor, Scalar boxColor, Scalar landColor);
};

