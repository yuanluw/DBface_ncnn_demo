#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>
#include<vector>

#include"detect.h"
#include"myutils.h"

using namespace std;
using namespace cv;



float THRESHOLD = 0.4;
float IOU = 0.5;


int main() {
	
	myutils util;
	
	Mat img = imread("F:\\course\\Android_Face_Verification\\DBFace-master\\datas\\selfie.jpg");
	cout << "raw shape: " << img.size() << endl;
	cout << "load model==>" << endl;
	detect d(THRESHOLD, IOU);
	cout << "model forward==>" << endl;
	vector<Obj> objs = d.getObjs(img);
	cout << "obj size:" << objs.size() << endl;
	cout << "draw bbox==>" << endl;
	for (int i = 0; i < objs.size(); i++) {
		util.drawBbox(img, objs[i], Scalar(255,0,0), Scalar(0,255,0),Scalar(0,0,255));
	}
	imshow("test", img);
	imwrite("test.jpg", img);
	waitKey(0);
	destroyAllWindows();
	return 0;
}