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
            std::string nombre(namelist[i]->d_name);
            auto ind_extension = nombre.find(".jpg");
            if (ind_extension != std::string::npos) {
                std::string ruta = std::string("./Imagenes/") + nombre;
                cv::Mat imagen = cv::imread(ruta);
                if (imagen.empty()) continue;
                nombre.erase(ind_extension);
                nombre += ".png";
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


bool ordenar_por_area(const std::vector<cv::Point>& p1, const std::vector<cv::Point>& p2) {
    return contourArea(p1) < contourArea(p2);
}

// Para extraer la región de la cabeza, necesito sacar el contorno con segunda mayor área, pero que esta área sea mayor que un valor determinado y que cumpla la forma ovalada.
/*
Casos:
1. num_contours == 1: se descartan estas imágenes porque las cabezas están en los extremos

En los siguientes casos se establece un threshold de área = 5000.
2. num_contours == 2: se obtiene el contorno con segunda mayor área, que debe tener 
como mínimo area_threshold de área.
3. num_contours > 2: se obtiene el contorno con segunda mayor área, que debe tener 
como mínimo el mayor área entre area_threshold y el área del contorno con tercera mayor área. 
*/


void deteccion(cv::Mat& imagen, const std::string& nombre_imagen, std::vector<std::vector<cv::Point> >& contours) {
    SimpleBlobDetector::Params params;

    // params.filterByCircularity = true;
    // params.minCircularity = 0.5;

    double mayorArea = 0.0;
    double menorArea = 1e9;

    int num_contours = contours.size();
    debug(num_contours);
    if (num_contours == 1) return; // Ignorar las cabezas en los extremos

    std::sort(contours.begin(), contours.end(), ordenar_por_area);

    double area_threshold = 5000.0;

    params.filterByArea = true;
    if (num_contours > 2) {
        params.minArea = std::max(contourArea(contours[num_contours-3]) + 1, area_threshold);
    }
    else {
        params.minArea = area_threshold;
    }
    params.maxArea = contourArea(contours[num_contours-2]) + 10;

    params.filterByConvexity = true;
    params.minConvexity = 0.1;

    params.filterByInertia = true;
    params.minInertiaRatio = 0.1;
    // params.maxInertiaRatio = 0.9;

    cv::Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

    cv::Mat contourImage(imagen.size(), CV_8UC1, cv::Scalar(0));
    cv::drawContours(contourImage, contours, -1, cv::Scalar(255));
    
    std::vector<KeyPoint> keypoints;
    detector->detect(contourImage, keypoints);

    if (keypoints.size() == 1) {
        Mat imagen_con_keypoints;
        drawKeypoints(contourImage, keypoints, imagen_con_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        Point2f punto_central = keypoints[0].pt;
        
        int contorno_cabeza = 0;
        for (int i = 0; i < contours.size(); i++) {
            int dentro = pointPolygonTest(contours[i], punto_central, false);
            if (dentro == 1) {
                // encontré el contorno necesario
                contorno_cabeza = i;
                break;
            }
        }
    
        cv::Mat imagenFinal(imagen.size(), CV_8UC1, cv::Scalar(0));
        cv::drawContours(imagenFinal, contours, contorno_cabeza, cv::Scalar(255), -1);
        std::string ruta = std::string("./Blobs/") + nombre_imagen;
        // cv::imwrite(ruta, imagen_con_keypoints);
        cv::imwrite(ruta, imagenFinal);


    }
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