#include "myutils.h"

vector<Obj> myutils::nms(vector<Obj> objs, float iou)
{
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

Mat myutils::pad(Mat img, int stride)
{
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

float myutils::getIou(Box a, Box b)
{	
	float aArea = (a.r - a.x + 1) * (a.b - a.y + 1);
	float bArea = (b.r - b.x + 1) * (b.b - b.y + 1);

	float x1 = max(a.x, b.x);
	float y1 = max(a.y, b.y);
	float x2 = min(a.r, b.r);
	float y2 = min(a.b, b.b);
	float w = max(0.0f, x2 - x1 + 1);
	float h = max(0.0f, y2 - y1 + 1);
	float area = w * h;


	float iou = area / (aArea + bArea - area);
	return iou;
}

float myutils::myExp(float v)
{
	float gate = 1;
	float base = exp(1);
	if (abs(v) < gate)
		return v * base;

	if (v > 0)
		return exp(v);
	else
		return -exp(-v);
}

void myutils::drawBbox(Mat& img, Obj o, Scalar textColor, Scalar boxColor, Scalar landColor)
{
	Box b = o.box;
	float score = o.score;
	Landmark lan = o.landmark;

	Point P1, P2;
	P1.x = b.x;
	P1.y = b.y;
	P2.x = b.r;
	P2.y = b.b;
	rectangle(img, P1, P2, boxColor, 2, 8);

	char destination[100];
	sprintf_s(destination, "%2.3f", score);
	string text = "score:" + (string)destination;
	putText(img, text, P1, 0, 0.4, textColor);

	for (int i = 0; i < 5; i++) {
		Point p;
		p.x = lan.x[i];
		p.y = lan.y[i];
		circle(img, p, 2, landColor, 1);
	}
}

//get ||vector||
float myutils::getMold(const vector<float>& vec) {   
	int n = vec.size();
	float sum = 0.0;
	for (int i = 0; i < n; ++i)
		sum += vec[i] * vec[i];
	return sqrt(sum);
}

//cosine similarity
float myutils::getSimilarity(const vector<float>& lhs, const vector<float>& rhs) {
	int n = lhs.size();
	assert(n == rhs.size());
	float tmp = 0.0;  
	for (int i = 0; i < n; ++i)
		tmp += lhs[i] * rhs[i];
	return tmp / (getMold(lhs) * getMold(rhs));
}
