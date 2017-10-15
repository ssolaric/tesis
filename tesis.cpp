#define debug(x) do { std::cout << #x << ": " << x << "\n"; } while (0)
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <cstring>

#include <dirent.h>
#include <stdlib.h>

#include <opencv2/opencv.hpp>
#include <openvdb/openvdb.h>
#include <openvdb/Types.h>
#include <openvdb/tools/RayIntersector.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/VolumeToMesh.h>

#include "auxiliar.h"

typedef openvdb::math::Ray<double> RayT;
typedef RayT::Vec3Type Vec3T;

const int BACKGROUND_VALUE = 1;

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

void tallar(const RayT& ray, openvdb::FloatGrid::Accessor& accessor, openvdb::tools::VolumeRayIntersector<openvdb::FloatGrid>& inter) {
    double t0 = 0;    
    double t1 = 0;    

    // while (inter.march(t0, t1)) {
    //     double step = 0.01;
    //     for (double t = t0; t <= t1; t += step) {
    //         Vec3T voxel = ray.eye() + ray.dir() * t;
    //         openvdb::Coord coordenada(voxel.x(), voxel.y(), voxel.z());
    //         accessor.setValueOff(coordenada, BACKGROUND_VALUE);
    //     }
    // }

    inter.march(t0, t1);
    double step = 0.01;
    for (double t = t0; t <= t1; t += step) {
        Vec3T voxel = ray.eye() + ray.dir() * t;
        openvdb::Coord coordenada(voxel.x(), voxel.y(), voxel.z());
        accessor.setValueOff(coordenada, BACKGROUND_VALUE);
    }
    
}

void reconstruccion_por_imagen(int orden, int num_imagenes, cv::Mat& imagen, openvdb::FloatGrid::Accessor& accessor, openvdb::tools::VolumeRayIntersector<openvdb::FloatGrid>& inter) {
    double theta = orden * 2*CV_PI / num_imagenes;
    double d = 200.0; // arbitrario, debe ser mayor que el lado del cubo entre sqrt(2)
    int l = imagen.rows;
    int h = imagen.cols;
    debug(l);
    debug(h);
    for (int x = 0; x < l; x++) {
        double alfa = atan(1/d * (l/2 - x));
        for (int y = 0; y < h; y++) {
            if (imagen.at<uchar>(y, x) == 0) {
                double z = y - h / 2;
                const Vec3T eye(d*cos(theta) - d*tan(alfa)*sin(theta), d*sin(theta) + d*tan(alfa)*cos(theta), z);

                // debug(eye);

                Vec3T dir = Vec3T(-cos(theta), -sin(theta), 0);
                const RayT ray(eye, dir); // rayo en index space
                if (inter.setIndexRay(ray)) {
                    tallar(ray, accessor, inter);
                }
            }
        }
    }
}

// http://www.openvdb.org/documentation/doxygen/namespaceopenvdb_1_1v4__0__2_1_1tools.html#a1cc2d4f60d561afd2fb248e386ca67d8

template<typename GridOrTreeType>
typename GridOrTreeType::Ptr limpiar_escena(GridOrTreeType& volume) {
    std::vector<typename GridOrTreeType::Ptr> segments;
    
    // separa el grid en varios grids o árboles distintos
    // segments: vector de árboles o grids ordenados en orden descendiente de vóxeles activos
    openvdb::tools::segmentActiveVoxels(volume, segments);

    // necesito quedarme con el árbol o grid que contenga el vóxel de coordenadas (0, 0, 0)
    const size_t num_segments = segments.size();
    for (int i = 0; i < num_segments; i++) {
        auto accessor = segments[i]->getAccessor();
        if (accessor.isValueOn(openvdb::Coord(0, 0, 0))) {
            // lo encontré, siempre lo encontrará
            return segments[i];
        }
    }
    // return GridOrTreeType::Ptr();
}

void guardar_malla(const std::vector<openvdb::Vec3s>& puntos, const std::vector<openvdb::Vec4I>& cuadrilateros) {
    std::ofstream archivo("malla.off");
    archivo << "OFF\n";
    archivo << puntos.size() << " " << cuadrilateros.size() << " 0\n";
    for (int i = 0; i < puntos.size(); i++) {
        archivo << puntos[i].x() << " " << puntos[i].y() << " " << puntos[i].z() << "\n";
    }

    for (int i = 0; i < cuadrilateros.size(); i++) {
        archivo << "4 " << cuadrilateros[i].x() << " " << cuadrilateros[i].y() << " " << cuadrilateros[i].z() << " " << cuadrilateros[i].w() << "\n";
    }
}

// en meshlab: si el objeto sale opaco, invertir las normales de los triángulos
void reconstruccion(std::vector<cv::Mat>& imagenes) {
    // 1. Construir el cubo
    openvdb::initialize();
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(BACKGROUND_VALUE); // 1 es el background value
    openvdb::CoordBBox caja(openvdb::Coord(-100, -100, -100), openvdb::Coord(100, 100, 100));
    grid->fill(caja, -1); // -1 es el valor de un voxel activo (ya no se mostrará en el vdb_view)

    auto accessor = grid->getAccessor();
    openvdb::tools::VolumeRayIntersector<openvdb::FloatGrid> inter(*grid);    
    // 2. Tallar el cubo
    for (int i = 0; i < imagenes.size(); i++) {
        reconstruccion_por_imagen(i, imagenes.size(), imagenes[i], accessor, inter);
    }

    // 3. Limpiar la escena
    openvdb::FloatGrid::Ptr espermatozoide = limpiar_escena(*grid);

    // 4. Pasarla a malla.
    // Así ya no tendría que usar el visor del openvdb.
    std::vector<openvdb::Vec3s> puntos;
    std::vector<openvdb::Vec4I> cuadrilateros;
    openvdb::tools::volumeToMesh(*espermatozoide, puntos, cuadrilateros);
    debug(puntos.size());
    debug(cuadrilateros.size());
    guardar_malla(puntos, cuadrilateros);
}

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
    auto itIni = imagenesOE2.begin();
    std::vector<cv::Mat> imagenesOE3(itIni, itIni + 10); // coger las 10 primeras imágenes por mientras
    reconstruccion(imagenesOE3);
}
