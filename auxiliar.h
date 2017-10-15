#pragma once

#include <opencv2/opencv.hpp>
#include <cmath>

// https://stackoverflow.com/a/15009815
cv::Mat equalizeIntensity(const cv::Mat& inputImage) {
    if(inputImage.channels() >= 3) {
        cv::Mat ycrcb;
        cv::cvtColor(inputImage, ycrcb, CV_BGR2YCrCb);

        std::vector<cv::Mat> channels;
        cv::split(ycrcb, channels);

        cv::equalizeHist(channels[0], channels[0]);

        cv::Mat result;
        cv::merge(channels,ycrcb);
        cv::cvtColor(ycrcb,result,CV_YCrCb2BGR);
        return result;
    }
    return cv::Mat();
}

bool ordenarPorArea(const std::vector<cv::Point>& p1, const std::vector<cv::Point>& p2) {
    return cv::contourArea(p1) < cv::contourArea(p2);
}

// http://docs.opencv.org/3.1.0/d1/dee/tutorial_introduction_to_pca.html
void drawAxis(cv::Mat& img, cv::Point p, cv::Point q, cv::Scalar colour, const float scale = 0.2) {
    double angle = std::atan2(1.0 * (p.y - q.y), 1.0 * (p.x - q.x)); // angle in radians
    double hypotenuse = std::sqrt((1.0 * (p.y - q.y) * (p.y - q.y) + (p.x - q.x) * (p.x - q.x)));
    // double degrees = angle * 180 / CV_PI; // convert radians to degrees (0-180 range)
    // cout << "Degrees: " << abs(degrees - 180) << endl; // angle in 0-360 degrees range
    // Here we lengthen the arrow by a factor of scale
    q.x = static_cast<int>(p.x - scale * hypotenuse * std::cos(angle));
    q.y = static_cast<int>(p.y - scale * hypotenuse * std::sin(angle));
    cv::line(img, p, q, colour, 1, CV_AA);
    // create the arrow hooks
    p.x = static_cast<int>(q.x + 9 * std::cos(angle + CV_PI / 4));
    p.y = static_cast<int>(q.y + 9 * std::sin(angle + CV_PI / 4));
    cv::line(img, p, q, colour, 1, CV_AA);
    p.x = static_cast<int>(q.x + 9 * std::cos(angle - CV_PI / 4));
    p.y = static_cast<int>(q.y + 9 * std::sin(angle - CV_PI / 4));
    cv::line(img, p, q, colour, 1, CV_AA);
}

double getOrientation(const std::vector<cv::Point> &pts, cv::Mat &img) {
    //Construct a buffer used by the pca analysis
    int sz = static_cast<int>(pts.size());
    cv::Mat dataPts = cv::Mat(sz, 2, CV_64FC1);
    for (int i = 0; i < dataPts.rows; ++i) {
        dataPts.at<double>(i, 0) = pts[i].x;
        dataPts.at<double>(i, 1) = pts[i].y;
    }
    //Perform PCA analysis
    cv::PCA pcaAnalysis(dataPts, cv::Mat(), CV_PCA_DATA_AS_ROW);
    //Store the center of the object
    cv::Point centro = cv::Point(static_cast<int>(pcaAnalysis.mean.at<double>(0, 0)), static_cast<int>(pcaAnalysis.mean.at<double>(0, 1)));
    //Store the eigenvalues and eigenvectors
    std::vector<cv::Point2d> eigenVecs(2);
    std::vector<double> eigenVals(2);
    for (int i = 0; i < 2; ++i) {
        eigenVecs[i] = cv::Point2d(pcaAnalysis.eigenvectors.at<double>(i, 0), pcaAnalysis.eigenvectors.at<double>(i, 1));
        eigenVals[i] = pcaAnalysis.eigenvalues.at<double>(0, i);
    }
    // Draw the principal components
    // Comenté esto para ya no mostrar los ejes
    // cv::circle(img, centro, 3, cv::Scalar(255, 0, 255), 2);
    // cv::Point p1 = centro + 0.02 * cv::Point(static_cast<int>(eigenVecs[0].x * eigenVals[0]), static_cast<int>(eigenVecs[0].y * eigenVals[0]));
    // cv::Point p2 = centro - 0.02 * cv::Point(static_cast<int>(eigenVecs[1].x * eigenVals[1]), static_cast<int>(eigenVecs[1].y * eigenVals[1]));
    // cv::drawAxis(img, centro, p1, cv::Scalar(0, 255, 0), 1);
    // cv::drawAxis(img, centro, p2, cv::Scalar(255, 255, 0), 5);
    double angle = std::atan2(eigenVecs[0].y, eigenVecs[0].x); // orientation in radians
    return angle;
}

// https://stackoverflow.com/a/24352524
void rotarImagen(cv::Mat& imagen, double angulo) {
    cv::Point2f centro(imagen.cols / 2.0, imagen.rows / 2.0);
    cv::Mat rotacion = cv::getRotationMatrix2D(centro, angulo, 1.0); // este ángulo es en grados

    cv::Rect bbox = cv::RotatedRect(centro, imagen.size(), angulo).boundingRect();

    rotacion.at<double>(0, 2) += bbox.width/2.0 - centro.x;
    rotacion.at<double>(1, 2) += bbox.height/2.0 - centro.y;

    cv::warpAffine(imagen, imagen, rotacion, bbox.size());
}