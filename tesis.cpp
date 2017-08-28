#define debug(x) do { std::cout << #x << ": " << x << "\n"; } while (0)
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <regex>

#include <dirent.h>
#include <stdlib.h>

#include <opencv2/opencv.hpp>

// https://github.com/opencv/opencv_attic/blob/master/opencv/samples/cpp/meanshift_segmentation.cpp

using namespace cv;
void floodFillPostprocess( Mat& img, const Scalar& colorDiff=Scalar::all(1) )
{
    CV_Assert( !img.empty() );
    RNG rng = theRNG();
    Mat mask( img.rows+2, img.cols+2, CV_8UC1, Scalar::all(0) );
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            if( mask.at<uchar>(y+1, x+1) == 0 )
            {
                Scalar newVal( rng(256), rng(256), rng(256) );
                floodFill( img, mask, Point(x,y), newVal, 0, colorDiff, colorDiff );
            }
        }
    }
}

// https://stackoverflow.com/a/15009815

Mat equalizeIntensity(const Mat& inputImage)
{
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

void leer_imagenes(std::vector<std::string>& nombresImagenes, std::vector<cv::Mat>& imagenes) {
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
                nombresImagenes.push_back(nombre);
            }
            free(namelist[i]);
        }
        free(namelist);
    }
}

void procesar_imagen(cv::Mat& imagen, const std::string& nombreImagen) {
    imagen = equalizeIntensity(imagen); // ecualización del canal de intensidad 
    double spatialWindowRadius = 2; // sp
    double colorWindowRadius = 5; // sr
    cv::pyrMeanShiftFiltering(imagen, imagen, spatialWindowRadius, colorWindowRadius);
    //floodFillPostprocess(imagen, Scalar::all(2));
    // 2. Aplicar Median Blur a la imagen en escala de grises.
    cv::cvtColor(imagen, imagen, cv::COLOR_BGR2GRAY);
    for (int j = 1; j < 11; j += 2) {
        // cv::medianBlur(imagen, imagen, j);
        cv::GaussianBlur(imagen, imagen, Size(j, j), 0);
    }
    
    // 3. Hacer un resize
    // ¿Conviene o no conviene?
    // Conviene: si el resize es bueno, permite mayor detalle para la reconstrucción.
    // No conviene: ya no cogería la forma original del espermatozoide.
    cv::resize(imagen, imagen, cv::Size(0, 0), 8.0, 8.0, cv::INTER_CUBIC);

    // 4. Aumentar el contraste
    cv::equalizeHist(imagen, imagen);
    // 5. Hacer el threshold

    // El área oscura dentro de un espermatozoide (los píxeles entre 0 y 2) es la parte inferior de la cabeza (cercana a la cola).
    // No funciona bien con la imagen 14 (posible opción: descartarla)
    cv::inRange(imagen, cv::Scalar(3), cv::Scalar(25), imagen);
    
    // cv::adaptiveThreshold(imagen, imagen, 255, ADAPTIVE_THRESH_MEAN_C,  THRESH_BINARY_INV, 3, 3);
    // cv::threshold(imagen, imagen, 40, 255, THRESH_BINARY);
    std::string ruta = std::string("./ImagenesProcesadas/") + nombreImagen;
    cv::imwrite(ruta, imagen);
}


// Según Francisco, las imágenes que no tienen out son luego de aplicar su clasificador. Es decir, trabajaré con estas (del 1 al 100).

int main() {
    std::vector<std::string> nombresImagenes;
    std::vector<cv::Mat> imagenes;
    leer_imagenes(nombresImagenes, imagenes);

    // for (size_t i = 0; i < imagenes.size(); i++) {
    for (size_t i = 0; i < imagenes.size(); i++) {
        // 1. Aplicar Mean Shift a la imagen a color.

        cv::Mat imagen = quitar_margen(imagenes[i], 4);
        procesar_imagen(imagen, nombresImagenes[i]);
        // cv::namedWindow(nombresImagenes[i], cv::WINDOW_AUTOSIZE);
        // cv::imshow(nombresImagenes[i], imagen);
        // cv::waitKey(2000);
        // // cv::waitKey(5000);
    }
}

// TODO: preguntar cómo hacer la segmentación de las cabezas y preparar la normalización de pose