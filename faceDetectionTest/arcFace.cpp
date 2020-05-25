#include "arcFace.h"

arcFace::arcFace()
{
	net.load_param("arcFace.param");
	net.load_model("arcFace.bin");
}

vector<float> arcFace::getFeature(Mat img)
{
	ncnn::Mat in = norm(img);
	ncnn::Extractor ex = net.create_extractor();//forward
	ex.input("0", in);
	ex.set_num_threads(4);
	ncnn::Mat feature;
	ex.extract("feature", feature);
	vector<float> f;
	float* ptr = feature.channel(0);
	for (int i = 0; i < feature.w; i++) {
		f.push_back(ptr[i]);
	}

	return f;
}

ncnn::Mat arcFace::norm(Mat img)
{
	ncnn::Mat in = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows);
	// normalization
	const float mean_vals_1[3] = { 0.485f * 255.0f , 0.456f * 255.0f, 0.406f * 255.0f };
	const float norm_vals_1[3] = { 1.0f / 0.229f / 255.0f, 1.0f / 0.224f / 255.0f, 1.0f / 0.225f / 255.0f };
	in.substract_mean_normalize(mean_vals_1, norm_vals_1);
	return in;
}