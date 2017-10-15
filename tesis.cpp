#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <cstring>

#include <opencv2/opencv.hpp>

#include "auxiliar.h"
#include "oe1y2.h"
// #include "oe3.h"

int main() {
    std::vector<std::string> nombresImagenesOE1;
    std::vector<cv::Mat> imagenesOE1;
    leerImagenes(nombresImagenesOE1, imagenesOE1);
    std::vector<std::vector<cv::Point>> contornosOE1;

    std::vector<std::string> nombresImagenesOE2;
    std::vector<cv::Mat> imagenesOE2;
    std::vector<std::vector<cv::Point>> contornosOE2;
    
    // OE1: Mejora de contraste y segmentación
    for (size_t i = 0; i < imagenesOE1.size(); i++) {
        imagenesOE1[i] = quitarMargen(imagenesOE1[i], 4);
        binarizacion(imagenesOE1[i], nombresImagenesOE1[i]);
        contorno(imagenesOE1[i], nombresImagenesOE1[i], contornosOE1);
        int indContorno = 0;
        bool eliminarImagen = deteccion(imagenesOE1[i], nombresImagenesOE1[i], contornosOE1, indContorno);

        if (!eliminarImagen) {
            imagenesOE2.push_back(imagenesOE1[i].clone());
            nombresImagenesOE2.push_back(nombresImagenesOE1[i]);
            contornosOE2.push_back(contornosOE1[indContorno]);
        }
    }

    // OE2: Normalización de pose
    for (size_t i = 0; i < imagenesOE2.size(); i++) {
        normalizacionDePose(imagenesOE2[i], nombresImagenesOE2[i], contornosOE2[i]);
    }

    // OE3: Reconstrucción
    // auto itIni = imagenesOE2.begin();
    // std::vector<cv::Mat> imagenesOE3(itIni, itIni + 10); // coger las 10 primeras imágenes por mientras
    // reconstruccion(imagenesOE3);
}
