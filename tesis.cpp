#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include "auxiliar.h"
#include "oe1y2.h"
#include "oe3.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Uso: " << argv[0] << " <num_imagenes> <step> <¿manual?>\n";
        std::exit(1);
    }

    int numImagenesReconstruccion = std::stoi(argv[1], nullptr);
    double step = std::stod(argv[2], nullptr);
    

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

    if (argv[3] != NULL && std::strcmp(argv[3], "manual") == 0) {
        // leer imagenes de la carpeta ImagenesReconstruccion
        std::cout << "Se procesarán las imágenes de la carpeta ImagenesReconstruccion.\n";
        std::vector<std::string> nombresImagenesOE3;
        std::vector<cv::Mat> imagenesOE3Aux;
        leerImagenes(nombresImagenesOE3, imagenesOE3Aux, true);
        numImagenesReconstruccion = std::min(numImagenesReconstruccion, int(imagenesOE3Aux.size()));
        debug(numImagenesReconstruccion);
        auto itIni = imagenesOE3Aux.begin();
        std::vector<cv::Mat> imagenesOE3(itIni, itIni + numImagenesReconstruccion);

        std::string nombreMalla = "malla_" + std::to_string(numImagenesReconstruccion) + "_imagenes_step_" + std::to_string(step);
        reconstruccion(imagenesOE3, step, nombreMalla);
    }
    else {
        std::cout << "Se procesarán las imágenes en memoria.\n";
        numImagenesReconstruccion = std::min(numImagenesReconstruccion, int(imagenesOE2.size()));
        auto itIni = imagenesOE2.begin();
        std::vector<cv::Mat> imagenesOE3(itIni, itIni + numImagenesReconstruccion);

        std::string nombreMalla = "malla_" + std::to_string(numImagenesReconstruccion) + "imagenes_step" + std::to_string(step);        
        reconstruccion(imagenesOE3, step, nombreMalla);
    }
    
}
