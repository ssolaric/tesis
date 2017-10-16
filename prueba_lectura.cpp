#define debug(x) do { std::cout << #x << ": " << x << "\n"; } while (0)
#include <iostream>
#include <vector>
#include <string>
#include <cstring>

#include <opencv2/opencv.hpp>
using namespace cv;


int main() {
    cv::Mat image = cv::imread("./ImagenesReconstruccion/1.png");
    debug(image.cols);
    debug(image.rows);
}

// TODO: preguntar cómo hacer la segmentación de las cabezas y preparar la normalización de pose