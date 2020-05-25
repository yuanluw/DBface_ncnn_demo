#include<iostream>
#include <fstream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>
#include<vector>

#include"detect.h"
#include"myutils.h"
#include"arcFace.h"

using namespace std;
using namespace cv;


float THRESHOLD = 0.4;
float IOU = 0.5;

myutils util;


float getSimilarity(vector<Obj> objs, Mat img, int flag);



int main() {

	cout << "load model==>" << endl;
	detect d(THRESHOLD, IOU);
	clock_t startTime, endTime;
	VideoCapture capture(0);

	while (true) {
		Mat frame;
		capture >> frame;
		cout << frame.size << endl;
		startTime = clock();
		vector<Obj> objs = d.getObjs(frame);
		endTime = clock();
		cout << "Totle Time : " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
		//cout << "obj size:" << objs.size() << endl;
		//cout << "draw bbox==>" << endl;
		for (int i = 0; i < objs.size(); i++) {
			util.drawBbox(frame, objs[i], Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255));
		}

		imshow("read", frame);

		waitKey(1);
	}

	
	//G:\\Ñ¸À×ÏÂÔØ\\Mul.ti-Task Fac.al Lan.dmark (MT.FL) dat.a.set\\AFLW\\0002-0926-image28054.jpg
	Mat img = imread("matt.JPG");
	cout << "raw shape: " << img.size() << endl;

	cout << "model forward==>" << endl;
	vector<Obj> objs = d.getObjs(img);
	cout << "obj size:" << objs.size() << endl;
	float similarity;
	if (objs.size() == 1)
		similarity = getSimilarity(objs, img, 0);
	
	
	//imwrite("detectFace.jpg", img);
	cout << "end============>" << endl;
	waitKey(0);
	destroyAllWindows();
	return 0;
}


float getSimilarity(vector<Obj> objs, Mat img, int flag) {
	
	Box b = objs[0].box;
	int x = max(b.x - img.size[0] / 10, 0);
	int y = max(b.y - img.size[1] / 10, 0);
	int w = min(b.r - x + img.size[0] / 10, img.size[0]);
	int h = min(b.b - y + img.size[1] / 10, img.size[1]);
	Rect rect(x, y, w, h);
	Mat imgRoi = img(rect);
	resize(imgRoi, imgRoi, Size(128, 128));
	cout << imgRoi.size() << endl;
	imshow("test", imgRoi);
	arcFace featureNet;

	vector<float> feature = featureNet.getFeature(imgRoi);
	if (flag) {
		ofstream featureDict;
		featureDict.open("featuredict.txt", ios::app | ios::in);
		for (int i = 0; i < feature.size(); i++) {
			featureDict << feature[i] << " ";
		}
		featureDict << endl;
		return 0.0;
	}
	else {
		fstream featureDict;
		featureDict = fstream("featuredict.txt", ios::in);
		vector<float> saveFeature;
		float temp;
		for (int i = 0; i < 512; i++) {
			featureDict >> temp;
			saveFeature.push_back(temp);
		}
		float similarity = util.getSimilarity(feature, saveFeature);
		cout << similarity << endl;
		return similarity;
	}
}