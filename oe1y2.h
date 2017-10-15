#pragma once

#include <iostream>

#include <dirent.h>
#include <stdlib.h>

#include <opencv2/opencv.hpp>

void leerImagenes(std::vector<std::string>& nombresImagenes, std::vector<cv::Mat>& imagenes) {
    struct dirent** namelist;
    int n = scandir("./Imagenes", &namelist, NULL, versionsort);
    if (n < 0) {
        perror("scandir");
    }
    else {
        for (int i = 0; i < n; i++) {
            std::string nombre(namelist[i]->d_name);
            auto indExtension = nombre.find(".jpg");
            if (indExtension != std::string::npos) {
                std::string ruta = std::string("./Imagenes/") + nombre;
                cv::Mat imagen = cv::imread(ruta);
                if (imagen.empty()) continue;
                nombre.erase(indExtension);
                nombre += ".png";
                imagenes.push_back(imagen);
                nombresImagenes.push_back(nombre);
            }
            free(namelist[i]);
        }
        free(namelist);
    }
}

cv::Mat quitarMargen(const cv::Mat& imagen, int margen) {
    cv::Mat nueva(imagen.rows - 2*margen, imagen.cols - 2*margen, imagen.type());
    for (int r = margen; r < imagen.rows - margen; r++) {
        for (int c = margen; c < imagen.cols - margen; c++) {
            nueva.at<cv::Vec3b>(r-margen, c-margen) = imagen.at<cv::Vec3b>(r, c);
        }
    }
    return nueva;
}

void binarizacion(cv::Mat& imagen, const std::string& nombreImagen) {

    imagen = equalizeIntensity(imagen); // ecualización del canal de intensidad 
    double spatialWindowRadius = 2; // sp
    double colorWindowRadius = 5; // sr

    // 1. Aplicar Mean Shift a la imagen a color.
    cv::pyrMeanShiftFiltering(imagen, imagen, spatialWindowRadius, colorWindowRadius);

    // 2. Aplicar Gaussian Blur a la imagen en escala de grises.
    cv::cvtColor(imagen, imagen, cv::COLOR_BGR2GRAY);
    for (int j = 1; j < 11; j += 2) {
        cv::GaussianBlur(imagen, imagen, cv::Size(j, j), 0);
    }
    
    // 3. Hacer un resize
    cv::resize(imagen, imagen, cv::Size(0, 0), 8.0, 8.0, cv::INTER_CUBIC);

    // 4. Aumentar el contraste
    cv::equalizeHist(imagen, imagen);

    // 5. Hacer el threshold
    // El área oscura dentro de un espermatozoide (los píxeles entre 0 y 2) es la parte inferior de la cabeza (cercana a la cola).
    // No funciona bien con la imagen 14 (posible opción: descartarla)
    // inRange(imagen, cv::Scalar(3), cv::Scalar(25), imagen);
    cv::inRange(imagen, cv::Scalar(0), cv::Scalar(25), imagen);
    cv::bitwise_not(imagen, imagen);

    std::string ruta = std::string("./ImagenesBinarizadas/") + nombreImagen;
    cv::imwrite(ruta, imagen);
}

// https://stackoverflow.com/a/8467129
void contorno(cv::Mat& imagen, const std::string& nombreImagen, std::vector<std::vector<cv::Point> >& contours) {
    // cv::Mat contourOutput = imagen.clone();
    cv::findContours(imagen, contours, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

    //Draw the contours
    cv::Mat contourImage(imagen.size(), CV_8UC3, cv::Scalar(0,0,0));
    cv::Scalar colors[3];
    colors[0] = cv::Scalar(255, 0, 0);
    colors[1] = cv::Scalar(0, 255, 0);
    colors[2] = cv::Scalar(0, 0, 255);
    cv::drawContours(contourImage, contours, -1, colors[1]);

    std::string ruta = std::string("./Contornos/") + nombreImagen;
    cv::imwrite(ruta, contourImage);
}

// Retorna true si hay que eliminar la imagen, false en caso contrario.
bool deteccion(cv::Mat& imagen, const std::string& nombreImagen, std::vector<std::vector<cv::Point> >& contours, int& contornoCabeza) {
    // Para extraer la región de la cabeza, necesito sacar el contorno con segunda mayor área, pero que esta área sea mayor que un valor determinado y que cumpla la forma ovalada.
    /*
    Casos:
    1. numContours == 1: se descartan estas imágenes porque las cabezas están en los extremos

    En los siguientes casos se establece un threshold de área = 5000.
    2. numContours == 2: se obtiene el contorno con segunda mayor área, que debe tener 
    como mínimo areaThreshold de área.
    3. numContours > 2: se obtiene el contorno con segunda mayor área, que debe tener 
    como mínimo el mayor área entre areaThreshold y el área del contorno con tercera mayor área. 
    */
    cv::SimpleBlobDetector::Params params;

    // params.filterByCircularity = true;
    // params.minCircularity = 0.5;

    double mayorArea = 0.0;
    double menorArea = 1e9;

    int numContours = contours.size();
    if (numContours == 1) return true; // Ignorar las cabezas en los extremos

    std::sort(contours.begin(), contours.end(), ordenarPorArea);

    double areaThreshold = 5000.0;
    params.filterByArea = true;
    if (numContours > 2) {
        params.minArea = std::max(cv::contourArea(contours[numContours-3]) + 1, areaThreshold);
    }
    else {
        params.minArea = areaThreshold;
    }
    params.maxArea = cv::contourArea(contours[numContours-2]) + 10;

    params.filterByConvexity = true;
    params.minConvexity = 0.1;

    params.filterByInertia = true;
    params.minInertiaRatio = 0.1;

    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);

    cv::Mat contourImage(imagen.size(), CV_8UC1, cv::Scalar(0));
    cv::drawContours(contourImage, contours, -1, cv::Scalar(255));
    
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(contourImage, keypoints);

    // Si encontré un solo keypoint, busco la región que contiene a este punto. Esta región es la cabeza del espermatozoide.
    // https://stackoverflow.com/a/30810250
    if (keypoints.size() == 1) {
        cv::Mat imagenConKeypoints;
        drawKeypoints(contourImage, keypoints, imagenConKeypoints, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        cv::Point2f puntoCentral = keypoints[0].pt;
        
        for (int i = 0; i < contours.size(); i++) {
            int dentro = pointPolygonTest(contours[i], puntoCentral, false);
            if (dentro == 1) {
                // encontré el contorno necesario
                contornoCabeza = i;
                break;
            }
        }

        // Grafico la región del keypoint enocntrado
        cv::Mat imagenFinal(imagen.size(), CV_8UC1, cv::Scalar(0));
        cv::drawContours(imagenFinal, contours, contornoCabeza, cv::Scalar(255), -1);
        std::string ruta = std::string("./Blobs/") + nombreImagen;
        cv::imwrite(ruta, imagenFinal);

        // guardar la imagen creada
        imagen = imagenFinal;
        return false;
    }
    else {
        return true;
    }
}

void normalizacionDePose(cv::Mat& imagen, const std::string& nombreImagen, const std::vector<cv::Point>& contorno) {
    cv::Rect rectangulo = cv::boundingRect(contorno);
    // cv::rectangle(imagen, rectangulo, cv::Scalar(255));
    // https://stackoverflow.com/a/8110695
    cv::Mat roi = cv::Mat(imagen, rectangulo).clone();

    double angulo = getOrientation(contorno, imagen); // en radianes
    angulo = angulo*180.0/CV_PI; // pasar a grados
    rotarImagen(roi, angulo); // ángulo en grados

    imagen = roi;

    std::string ruta = std::string("./NormalizadasSinEjes/") + nombreImagen;
    // cv::imwrite(ruta, imagen);
    cv::imwrite(ruta, roi);
}