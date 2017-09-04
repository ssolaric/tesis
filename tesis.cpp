#define debug(x) do { std::cout << #x << ": " << x << "\n"; } while (0)
#include <iostream>
#include <vector>
#include <string>
#include <cstring>

#include <dirent.h>
#include <stdlib.h>

#include <opencv2/opencv.hpp>
using namespace cv;

// https://stackoverflow.com/a/15009815
Mat equalizeIntensity(const Mat& inputImage) {
    if(inputImage.channels() >= 3)
    {
        Mat ycrcb;

        cvtColor(inputImage,ycrcb,CV_BGR2YCrCb);

        std::vector<Mat> channels;
        split(ycrcb,channels);

        equalizeHist(channels[0], channels[0]);

        Mat result;
        merge(channels,ycrcb);

        cvtColor(ycrcb,result,CV_YCrCb2BGR);

        return result;
    }
    return Mat();
}


cv::Mat quitar_margen(const cv::Mat& imagen, int margen) {
    cv::Mat nueva(imagen.rows - 2*margen, imagen.cols - 2*margen, imagen.type());
    for (int r = margen; r < imagen.rows - margen; r++) {
        for (int c = margen; c < imagen.cols - margen; c++) {
            nueva.at<cv::Vec3b>(r-margen, c-margen) = imagen.at<cv::Vec3b>(r, c);
        }
    }
    return nueva;
}

void leer_imagenes(std::vector<std::string>& nombres_imagenes, std::vector<cv::Mat>& imagenes) {
    struct dirent** namelist;
    int n = scandir("./Imagenes", &namelist, NULL, versionsort);
    if (n < 0) {
        perror("scandir");
    }
    else {
        for (int i = 0; i < n; i++) {
            const char* nombre = namelist[i]->d_name;
            if (std::strstr(nombre, ".jpg") != NULL) {
                std::string ruta = std::string("./Imagenes/") + nombre;
                cv::Mat imagen = cv::imread(ruta);
                if (imagen.empty()) continue;
                imagenes.push_back(imagen);
                nombres_imagenes.push_back(nombre);
            }
            free(namelist[i]);
        }
        free(namelist);
    }
}

// https://stackoverflow.com/a/8467129
void contorno(cv::Mat& imagen, const std::string& nombre_imagen, std::vector<std::vector<cv::Point> >& contours) {
    // cv::Mat contourOutput = imagen.clone();
    cv::findContours(imagen, contours, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

    //Draw the contours
    cv::Mat contourImage(imagen.size(), CV_8UC3, cv::Scalar(0,0,0));
    cv::Scalar colors[3];
    colors[0] = cv::Scalar(255, 0, 0);
    colors[1] = cv::Scalar(0, 255, 0);
    colors[2] = cv::Scalar(0, 0, 255);
    // for (size_t i = 0; i < contours.size(); i++) {
    //     cv::drawContours(contourImage, contours, i, colors[idx % 3]); // dibuja en contourImage el contorno contours[i]
    // }
    cv::drawContours(contourImage, contours, -1, colors[1]);

    std::string ruta = std::string("./Contornos/") + nombre_imagen;
    cv::imwrite(ruta, contourImage);
}

void binarizacion(cv::Mat& imagen, const std::string& nombre_imagen) {

    imagen = equalizeIntensity(imagen); // ecualización del canal de intensidad 
    double spatialWindowRadius = 2; // sp
    double colorWindowRadius = 5; // sr

    // 1. Aplicar Mean Shift a la imagen a color.
    cv::pyrMeanShiftFiltering(imagen, imagen, spatialWindowRadius, colorWindowRadius);

    // 2. Aplicar Gaussian Blur a la imagen en escala de grises.
    cv::cvtColor(imagen, imagen, cv::COLOR_BGR2GRAY);
    for (int j = 1; j < 11; j += 2) {
        cv::GaussianBlur(imagen, imagen, Size(j, j), 0);
    }
    
    // 3. Hacer un resize
    cv::resize(imagen, imagen, cv::Size(0, 0), 8.0, 8.0, cv::INTER_CUBIC);

    // 4. Aumentar el contraste
    cv::equalizeHist(imagen, imagen);

    // 5. Hacer el threshold
    // El área oscura dentro de un espermatozoide (los píxeles entre 0 y 2) es la parte inferior de la cabeza (cercana a la cola).
    // No funciona bien con la imagen 14 (posible opción: descartarla)
    // cv::inRange(imagen, cv::Scalar(3), cv::Scalar(25), imagen);
    cv::inRange(imagen, cv::Scalar(0), cv::Scalar(25), imagen);
    cv::bitwise_not(imagen, imagen);

    std::string ruta = std::string("./ImagenesBinarizadas/") + nombre_imagen;
    cv::imwrite(ruta, imagen);
}

// Para extraer la región de la cabeza, necesito sacar el contorno con segunda mayor área, pero que esta área sea mayor que un valor determinado y que cumpla la forma ovalada.
void deteccion(cv::Mat& imagen, const std::string& nombre_imagen, const std::vector<std::vector<cv::Point> >& contours) {
    SimpleBlobDetector::Params params;

    // params.filterByCircularity = true;
    // params.minCircularity = 0.5;

    double mayorArea = 0.0;
    double menorArea = 1e9;
    debug(contours.size());
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        mayorArea = std::max(mayorArea, area);
        menorArea = std::min(menorArea, area);
    }
    debug(menorArea);
    debug(mayorArea);
    cv::Mat contourImage(imagen.size(), CV_8UC3, cv::Scalar(0,0,0));
    

    params.filterByArea = true;
    params.minArea = menorArea;
    params.maxArea = mayorArea;

    params.filterByConvexity = true;
    params.minConvexity = 0.1;

    params.filterByInertia = true;
    params.minInertiaRatio = 0.3;
    // params.maxInertiaRatio = 0.9;

    cv::Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

    std::vector<KeyPoint> keypoints;
    detector->detect(imagen, keypoints);

    Mat imagen_con_keypoints;
    drawKeypoints(imagen, keypoints, imagen_con_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    std::string ruta = std::string("./Blobs/") + nombre_imagen;
    cv::imwrite(ruta, imagen_con_keypoints);

}

int main() {
    std::vector<std::string> nombresImagenes;
    std::vector<cv::Mat> imagenes;
    leer_imagenes(nombresImagenes, imagenes);
    for (size_t i = 0; i < imagenes.size(); i++) {
        imagenes[i] = quitar_margen(imagenes[i], 4);
        binarizacion(imagenes[i], nombresImagenes[i]);
        std::vector<std::vector<cv::Point> > contours;
        contorno(imagenes[i], nombresImagenes[i], contours);
        debug(i);
        deteccion(imagenes[i], nombresImagenes[i], contours);
    }


}

// TODO: preguntar cómo hacer la segmentación de las cabezas y preparar la normalización de pose