#include "detect.h"



detect::detect(float thresh, float iou)
{
	net.load_param("dbface16.param");
	net.load_model("dbface16.bin");
	THRESHOLD = thresh;
	IOU = iou;
}


vector<Obj> detect::getObjs(Mat img)
{
	img = util.pad(img);
	ncnn::Mat in = norm(img);
	ncnn::Extractor ex = net.create_extractor();//forward
	ex.input("0", in);
	//ex.set_num_threads(4);
	ncnn::Mat landmark, hm, hmPool, tlrb;
	ex.extract("landmark", landmark);
	ex.extract("hm", hm);
	ex.extract("pool_hm", hmPool);
	ex.extract("tlrb", tlrb);
	int hmWeight = hm.w;
	hm = hm.reshape(hm.c * hm.h * hm.w);
	hmPool = hmPool.reshape(hmPool.c * hmPool.w * hmPool.h);
	vector<Id> ids;
	//get suspected boxs
	genIds(hm, hmPool, hmWeight, THRESHOLD, ids);

	vector<Obj> objs;
	//get each box and key point information
	decode(hmWeight, ids, tlrb, landmark, objs);

	return util.nms(objs, IOU);
}

ncnn::Mat detect::norm(Mat img)
{
	ncnn::Mat in = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows);
	// normalization
	const float mean_vals_1[3] = { 0.485f * 255.0f , 0.456f * 255.0f, 0.406f * 255.0f };
	const float norm_vals_1[3] = { 1.0f / 0.229f / 255.0f, 1.0f / 0.224f / 255.0f, 1.0f / 0.225f / 255.0f };
	in.substract_mean_normalize(mean_vals_1, norm_vals_1);
	return in;
}

void detect::genIds(ncnn::Mat hm, ncnn::Mat hmPool, int w, double thresh, vector<Id>& ids)
{
	const float* ptr = hm.channel(0);
	const float* ptrPool = hmPool.channel(0);
	for (int i = 0; i < hm.w; i++) {

		float temp = 0.0;
		if ((ptr[i] - ptrPool[i]) < 0.01) {
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

void detect::decode(int w, vector<Id> ids, ncnn::Mat tlrb, ncnn::Mat landmark, vector<Obj> &objs)
{
	for (int i = 0; i < ids.size(); i++) {
		Obj objTemp;
		int cx = ids[i].idx;
		int cy = ids[i].idy;
		double score = ids[i].score;
		vector<float> boxTemp;
		//get each box information
		for (int j = 0; j < tlrb.c; j++) {
			const float* ptr = tlrb.channel(j);
			boxTemp.push_back(ptr[w * (cy - 1) + cx]);
		}
		objTemp.box.x = (cx - boxTemp[0]) * STRIDE;
		objTemp.box.y = (cy - boxTemp[1]) * STRIDE;
		objTemp.box.r = (cx + boxTemp[2]) * STRIDE;
		objTemp.box.b = (cy + boxTemp[3]) * STRIDE;
		objTemp.score = score;

		//get key point information
		Landmark lanTemp;
		for (int j = 0; j < 10; j++) {
			const float* ptr = landmark.channel(j);
			if (j < 5) {
				float temp = (util.myExp(ptr[w * (cy - 1) + cx] * 4) + cx) * STRIDE;
				lanTemp.x.push_back(temp);
			}
			else {
				float temp = (util.myExp(ptr[w * (cy - 1) + cx] * 4) + cy) * STRIDE;
				lanTemp.y.push_back(temp);
			}

		}
		objTemp.landmark = lanTemp;
		objs.push_back(objTemp);
	}
}


