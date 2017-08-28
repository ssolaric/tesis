#define debug(x) do { std::cout << #x << ": " << x << "\n"; } while (0)
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <regex>

#include <dirent.h>
#include <stdlib.h>

#include <opencv2/opencv.hpp>

cv::Mat quitar_margen(const cv::Mat& imagen, int margen) {
    cv::Mat nueva(imagen.rows - 2*margen, imagen.cols - 2*margen, imagen.type());
    for (int r = margen; r < imagen.rows - margen; r++) {
        for (int c = margen; c < imagen.cols - margen; c++) {
            nueva.at<cv::Vec3b>(r-margen, c-margen) =  imagen.at<cv::Vec3b>(r, c);
        }
    }
    return nueva;
}

int main() {
    struct dirent** namelist;
    int n = scandir("./Imagenes", &namelist, NULL, versionsort);
    std::vector<std::string> nombresImagenes;
    std::vector<cv::Mat> imagenes;
    
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
                nombresImagenes.push_back(nombre);
            }
            free(namelist[i]);
        }
        free(namelist);
    }

    // for (size_t i = 0; i < imagenes.size(); i++) {
    for (size_t i = 0; i < imagenes.size(); i++) {
        // 1. Aplicar Mean Shift a la imagen a color.

        cv::Mat imagen = quitar_margen(imagenes[i], 4);
        cv::imshow(nombresImagenes[i], imagen);
        cv::waitKey();

        double spatialWindowRadius = 2; // sp
        double colorWindowRadius = 5; // sr
        // cv::pyrMeanShiftFiltering(imagen, imagen, spatialWindowRadius, colorWindowRadius);
        // debug(imagen.rows); // height
        // debug(imagen.cols); // width

        // // 2. Aplicar Median Blur a la imagen en escala de grises.
        // cv::cvtColor(imagen, imagen, cv::COLOR_BGR2GRAY);
        // for (int j = 1; j < 11; j += 2) {
        //     cv::medianBlur(imagen, imagen, j);
        // }

        // // 3. Hacer un resize
        // cv::resize(imagen, imagen, cv::Size(0, 0), 8.0, 8.0, cv::INTER_CUBIC);

        // // 4. Aumentar el contraste
        // cv::equalizeHist(imagen, imagen);

        // 5. Hacer el threshold
        // cv::inRange(imagen, cv::Scalar(0), cv::Scalar(20), imagen);

        // cv::imshow(nombresImagenes[i], imagen);
        // cv::waitKey();
        // cv::waitKey(5000);
    }
}