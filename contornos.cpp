#define debug(x) do { std::cout << #x << ": " << x << "\n"; } while (0)
#include <iostream>
#include <vector>
#include <string>
#include <cstring>

#include <opencv2/opencv.hpp>
using namespace cv;


int main() {
    cv::Mat image = cv::imread("./ImagenesBinarizadas/1.jpg", 0);

    std::vector<std::vector<cv::Point> > contours;
    cv::Mat contourOutput = image.clone();
    cv::findContours( contourOutput, contours, CV_RETR_TREE, CV_CHAIN_APPROX_NONE );

    //Draw the contours
    cv::Mat contourImage(image.size(), CV_8UC3, cv::Scalar(0,0,0));
    cv::Scalar colors[3];
    colors[0] = cv::Scalar(255, 0, 0);
    colors[1] = cv::Scalar(0, 255, 0);
    colors[2] = cv::Scalar(0, 0, 255);
    for (size_t idx = 0; idx < contours.size(); idx++) {
        cv::drawContours(contourImage, contours, idx, colors[idx % 3]);
    }

    cv::imshow("Input Image", image);
    cvMoveWindow("Input Image", 0, 0);
    cv::imshow("Contours", contourImage);
    cvMoveWindow("Contours", 200, 0);
    cv::waitKey(0);
}

// TODO: preguntar cómo hacer la segmentación de las cabezas y preparar la normalización de pose