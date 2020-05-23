#include<iostream>
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

float THRESHOLD = 0.4;
float IOU = 0.5;

Mat pad(Mat img, int stride = 32) {
	bool hasChange = false;
	int stdw = img.cols;
	if (stdw % stride != 0) {
		stdw += stride - (stdw % stride);
		hasChange = true;
	}
	int stdh = img.rows;
	if (stdh % stride != 0) {
		stdh += stride - (stdh % stride);
		hasChange = true;
	}
	if (hasChange) {
		Mat newImg = Mat::zeros(stdh, stdw, CV_8UC3);
		Rect roi = Rect(0, 0, img.cols, img.rows);
		img.copyTo(newImg(roi));
		return newImg;
	}
	return img;


}

float getIou(Box a, Box b) {

	float aArea = (a.r - a.x + 1) * (a.b - a.y + 1);
	float bArea = (b.r - b.x + 1) * (b.b - b.y + 1);

	float x1 = max(a.x, b.x);
	float y1 = max(a.y, b.y);
	float x2 = min(a.r, b.r);
	float y2 = min(a.b, b.b);
	float w = max(0, x2 - x1 + 1);
	float h = max(0, y2 - y1 + 1);
	float area = w * h;


	float iou = area / (aArea + bArea - area);
	return iou;
}

vector<Obj> nms(vector<Obj> objs, float iou = 0.5) {

	if (objs.size() == 0) {
		return objs;
	}
	sort(objs.begin(), objs.end(), [](Obj a, Obj b) { return a.score < b.score; });

	vector<Obj> keep;
	int* flag = new int[objs.size()]();
	for (int i = 0; i < objs.size(); i++) {
		if (flag[i] != 0)
			continue;
		keep.push_back(objs[i]);
		for (int j = i + 1; j < objs.size(); j++) {
			if (flag[j] == 0 && getIou(objs[i].box, objs[j].box) > iou)
				flag[j] = 1;
		}
	}
	return keep;
}

float myExp(float v) {
	float gate = 1;
	float base = exp(1);
	if (abs(v) < gate)
		return v * base;

	if (v > 0)
		return exp(v);
	else
		return -exp(-v);
}

void genIds(ncnn::Mat hm, ncnn::Mat hmPool, int w, double thresh, vector<Id>& ids) {

	const float* ptr = hm.channel(0);

	for (int i = 0; i < hm.w; i++) {
		
		float temp = 0.0;
		if ((ptr[i] - ptr[i])<0.01) {
			temp = ptr[i];
		}

		if (ptr[i] > thresh) {
			Id temp;
			temp.idx = i % w;
			temp.idy = (int)(i / w);
			temp.score = ptr[i];
			ids.push_back(temp);
		}
	}
}

void drawBbox(Mat& img, Box b, double score, Landmark lan) {

	Point P1, P2;
	P1.x = b.x;
	P1.y = b.y;
	P2.x = b.r;
	P2.y = b.b;
	rectangle(img, P1, P2, Scalar(0, 255, 0), 2, 8);
	
	char destination[100];
	sprintf_s(destination, "%2.3f", score);
	string text = "score:" + (string)destination;
	putText(img, text, P1, 0, 0.4, Scalar(255, 0, 0));

	for (int i = 0; i < 5; i++) {
		Point p;
		p.x = lan.x[i];
		p.y = lan.y[i];
		circle(img, p, 2, Scalar(0, 0, 255), 1);
	}

}

int main() {
	
	Mat img = imread("F:\\course\\Android_Face_Verification\\DBFace-master\\datas\\selfie.jpg");
	//ensure image shape//32 == 0
	img = pad(img);
	cout << "raw shape: " << img.size() << endl;
	
	cout << "load model==>" << endl;
	ncnn::Net net;
	net.load_param("dbface.param");
	net.load_model("dbface.bin");

	ncnn::Mat in = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows);
    // normalization
	const float mean_vals_1[3] = { 0.485f * 255.0f , 0.456f * 255.0f, 0.406f * 255.0f };
	const float norm_vals_1[3] = { 1.0f / 0.229f / 255.0f, 1.0f / 0.224f / 255.0f, 1.0f / 0.225f / 255.0f };
	in.substract_mean_normalize(mean_vals_1, norm_vals_1);
	cout << "model forward==>" << endl;
	ncnn::Extractor ex = net.create_extractor();//forward
	ex.input("0", in);
	ex.set_num_threads(4);
	ncnn::Mat landmark, hm, hmPool, tlrb;
	ex.extract("landmark", landmark);
	ex.extract("hm", hm);
	ex.extract("pool_hm", hmPool);
	ex.extract("tlrb", tlrb);
	cout << "post prepocess==>" << endl;
	int hmWeight = hm.w;
	int hmHeight = hm.h;
	int hwChannel = hm.c;
	hm = hm.reshape(hm.c * hm.h * hm.w);
	hmPool = hmPool.reshape(hmPool.c * hmPool.w * hmPool.h);

	vector<Id> ids;
	//get suspected boxs
	genIds(hm, hmPool, hmWeight, THRESHOLD, ids);
	cout <<"ids size:"<<ids.size() << endl;
	float stride = 4;
	vector<Obj> objs;
	for (int i = 0; i < ids.size(); i++) {
		Obj objTemp;
		int cx = ids[i].idx;
		int cy = ids[i].idy;
		double score = ids[i].score;
		vector<float> boxTemp;
		//get each box information
		for (int j = 0; j < tlrb.c; j++) {
			const float* ptr = tlrb.channel(j);
			boxTemp.push_back(ptr[hmWeight * (cy - 1) + cx]);
		}
		objTemp.box.x = (cx - boxTemp[0]) * stride;
		objTemp.box.y = (cy - boxTemp[1]) * stride;
		objTemp.box.r = (cx + boxTemp[2]) * stride;
		objTemp.box.b = (cy + boxTemp[3]) * stride;
		objTemp.score = score;
		
		//get key point information
		Landmark lanTemp;
		for (int j = 0; j < 10; j++) {
			const float* ptr = landmark.channel(j);
			if (j < 5) {
				float temp = (myExp(ptr[hmWeight * (cy - 1) + cx] * 4) + cx) * stride;
				lanTemp.x.push_back(temp);
			}
			else {
				float temp = (myExp(ptr[hmWeight * (cy - 1) + cx] * 4) + cy) * stride;
				lanTemp.y.push_back(temp);
			}
				
		}
		objTemp.landmark = lanTemp;
		objs.push_back(objTemp);
	}
	cout << "obj size:" << objs.size() << endl;
	objs = nms(objs, IOU);
	cout << "obj size(after nms):" << objs.size() << endl;
	cout << "draw bbox==>" << endl;
	for (int i = 0; i < objs.size(); i++) {
		drawBbox(img, objs[i].box, objs[i].score, objs[i].landmark);
	}
	imshow("test", img);
	imwrite("test.jpg", img);
	waitKey(0);
	destroyAllWindows();
	return 0;
}