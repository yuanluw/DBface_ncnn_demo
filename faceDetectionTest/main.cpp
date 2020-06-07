#include<iostream>
#include <fstream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>
#include<vector>
#include <windows.h>

#include"detect.h"
#include"myutils.h"
#include"arcFace.h"

using namespace std;
using namespace cv;

myutils util;

float THRESHOLD = 0.4;
float IOU = 0.5;

vector<Obj> objs;
volatile bool calculateFlag = false;
vector<float> saveFeature;

float getSimilarity(vector<Obj> objs, Mat img, int flag, arcFace featureNet);
DWORD WINAPI calculate(LPVOID lpParamter);
void readSaveFeature();


int main() {

	int choice = 0;
	cout << "input choice:" << endl;
	cout << "0 photo test" << endl;
	cout << "1 video test" << endl;
	cin >> choice;
	readSaveFeature();
	if (choice) {
		VideoCapture capture(0);
		Mat frame, tempFrame;
		
		while (true) {

			if (!calculateFlag) {
				if (objs.size() > 0) {
					for (int i = 0; i < objs.size(); i++)
						util.drawBbox(tempFrame, objs[i], Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255));
					imshow("detectFace.jpg", tempFrame);
				}
				capture >> tempFrame;
				calculateFlag = true;
				CreateThread(NULL, 0, calculate, &tempFrame, 0, NULL);
			}
			else {
				capture >> frame;
				imshow("read", frame);
			}
			waitKey(1);
		}
	}
	else {
		Mat img = imread("matt3.JPG");
		float similarity;
		detect* d = new detect(THRESHOLD, IOU);
		arcFace*featureNet = new arcFace();
		cout << "model forward==>" << endl;
		objs = d->getObjs(img);
		delete d;
		cout << "detect face size: " << objs.size() << endl;
		if (objs.size() == 1) {
			int flag = 0; //计算当前图片与已保存特征的相似度
			float simi = getSimilarity(objs, img, flag, *featureNet);
			cout << "sim" << simi << endl;
		}
		for (int i = 0; i < objs.size(); i++) {
			util.drawBbox(img, objs[i], Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255));
		}
		imwrite("detectFace.jpg", img);
		cout << "end============>" << endl;
	}
	

	destroyAllWindows();
	return 0;
}


//flag为1 代表计算特征向量并保存
//flag为0 代表计算相似度
float getSimilarity(vector<Obj> objs, Mat img, int flag, arcFace featureNet) {
	
	Box b = objs[0].box;
	cout << b.x << " " << b.y << " " << b.r << " " << b.b << endl;
	int x = max(b.x - img.size[1] / 10, 0);
	int y = max(b.y - img.size[0] / 10, 0);
	
	int w = min(b.r - x + img.size[1] / 10, img.size[1]-x-1);
	int h = min(b.b - y + img.size[0] / 10, img.size[0]-y-1);
	Rect rect(x, y, w, h);
	Mat imgRoi = img(rect);
	resize(imgRoi, imgRoi, Size(128, 128));
	
	vector<float> feature = featureNet.getFeature(imgRoi);
	if (flag) {
		ofstream featureDict;
		featureDict.open("featuredict.txt",  ios::in);
		for (int i = 0; i < feature.size(); i++) {
			featureDict << feature[i] << " ";
		}
		featureDict << endl;
		return 0.0;
	}
	else {
		float similarity = util.getSimilarity(feature, saveFeature);
		return similarity;
	}
}

//
DWORD WINAPI calculate(LPVOID lpParamter)
{
	float similarity;
	detect *d = new detect(THRESHOLD, IOU);
	arcFace *featureNet = new arcFace();
	clock_t startTime, endTime;
	Mat* frame = (Mat*)lpParamter;
	startTime = clock();
	objs = d->getObjs(*frame);
	endTime = clock();
	cout << "detect face Time : " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	delete d;
	startTime = clock();
	if (objs.size() == 1) {
		similarity = getSimilarity(objs, *frame, 0, *featureNet);
		cout << "similarity: " << similarity << endl;
	}
	endTime = clock();
	cout << "calculate similarity Time : " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	//线程执行完毕
	calculateFlag = false;
	return 0L;
}


void readSaveFeature()
{
	fstream featureDict;
	featureDict = fstream("featuredict.txt", ios::in);
	float temp;
	for (int i = 0; i < 512; i++) {
		featureDict >> temp;
		saveFeature.push_back(temp);
	}
}