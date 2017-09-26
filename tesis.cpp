#define debug(x) do { std::cout << #x << ": " << x << "\n"; } while (0)
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <cstring>

#include <dirent.h>
#include <stdlib.h>

#include <opencv2/opencv.hpp>
#include <openvdb/openvdb.h>
#include <openvdb/tools/RayIntersector.h>

#include "auxiliar.h"
using namespace cv;
using namespace openvdb;
typedef math::Ray<double>  RayT;
typedef RayT::Vec3Type Vec3T;

void leer_imagenes(std::vector<std::string>& nombres_imagenes, std::vector<Mat>& imagenes) {
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
                Mat imagen = imread(ruta);
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

Mat quitar_margen(const Mat& imagen, int margen) {
    Mat nueva(imagen.rows - 2*margen, imagen.cols - 2*margen, imagen.type());
    for (int r = margen; r < imagen.rows - margen; r++) {
        for (int c = margen; c < imagen.cols - margen; c++) {
            nueva.at<Vec3b>(r-margen, c-margen) = imagen.at<Vec3b>(r, c);
        }
    }
    return nueva;
}

void binarizacion(Mat& imagen, const std::string& nombre_imagen) {

    imagen = equalizeIntensity(imagen); // ecualización del canal de intensidad 
    double spatialWindowRadius = 2; // sp
    double colorWindowRadius = 5; // sr

    // 1. Aplicar Mean Shift a la imagen a color.
    pyrMeanShiftFiltering(imagen, imagen, spatialWindowRadius, colorWindowRadius);

    // 2. Aplicar Gaussian Blur a la imagen en escala de grises.
    cvtColor(imagen, imagen, COLOR_BGR2GRAY);
    for (int j = 1; j < 11; j += 2) {
        GaussianBlur(imagen, imagen, Size(j, j), 0);
    }
    
    // 3. Hacer un resize
    resize(imagen, imagen, Size(0, 0), 8.0, 8.0, INTER_CUBIC);

    // 4. Aumentar el contraste
    equalizeHist(imagen, imagen);

    // 5. Hacer el threshold
    // El área oscura dentro de un espermatozoide (los píxeles entre 0 y 2) es la parte inferior de la cabeza (cercana a la cola).
    // No funciona bien con la imagen 14 (posible opción: descartarla)
    // inRange(imagen, Scalar(3), Scalar(25), imagen);
    inRange(imagen, Scalar(0), Scalar(25), imagen);
    bitwise_not(imagen, imagen);

    std::string ruta = std::string("./ImagenesBinarizadas/") + nombre_imagen;
    imwrite(ruta, imagen);
}

// https://stackoverflow.com/a/8467129
void contorno(Mat& imagen, const std::string& nombre_imagen, std::vector<std::vector<Point> >& contours) {
    // Mat contourOutput = imagen.clone();
    findContours(imagen, contours, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

    //Draw the contours
    Mat contourImage(imagen.size(), CV_8UC3, Scalar(0,0,0));
    Scalar colors[3];
    colors[0] = Scalar(255, 0, 0);
    colors[1] = Scalar(0, 255, 0);
    colors[2] = Scalar(0, 0, 255);
    // for (size_t i = 0; i < contours.size(); i++) {
    //     drawContours(contourImage, contours, i, colors[idx % 3]); // dibuja en contourImage el contorno contours[i]
    // }
    drawContours(contourImage, contours, -1, colors[1]);

    std::string ruta = std::string("./Contornos/") + nombre_imagen;
    imwrite(ruta, contourImage);
}

// Retorna true si hay que eliminar la imagen, false en caso contrario.
bool deteccion(Mat& imagen, const std::string& nombre_imagen, std::vector<std::vector<Point> >& contours, int& contorno_cabeza) {
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
    SimpleBlobDetector::Params params;

    // params.filterByCircularity = true;
    // params.minCircularity = 0.5;

    double mayorArea = 0.0;
    double menorArea = 1e9;

    int num_contours = contours.size();
    if (num_contours == 1) return true; // Ignorar las cabezas en los extremos

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

    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

    Mat contourImage(imagen.size(), CV_8UC1, Scalar(0));
    drawContours(contourImage, contours, -1, Scalar(255));
    
    std::vector<KeyPoint> keypoints;
    detector->detect(contourImage, keypoints);

    // Si encontré un solo keypoint, busco la región que contiene a este punto. Esta región es la cabeza del espermatozoide.
    // https://stackoverflow.com/a/30810250
    if (keypoints.size() == 1) {
        Mat imagen_con_keypoints;
        drawKeypoints(contourImage, keypoints, imagen_con_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        Point2f punto_central = keypoints[0].pt;
        
        for (int i = 0; i < contours.size(); i++) {
            int dentro = pointPolygonTest(contours[i], punto_central, false);
            if (dentro == 1) {
                // encontré el contorno necesario
                contorno_cabeza = i;
                break;
            }
        }

        // Grafico la región del keypoint enocntrado
        Mat imagenFinal(imagen.size(), CV_8UC1, Scalar(0));
        drawContours(imagenFinal, contours, contorno_cabeza, Scalar(255), -1);
        std::string ruta = std::string("./Blobs/") + nombre_imagen;
        imwrite(ruta, imagenFinal);

        // guardar la imagen creada
        imagen = imagenFinal;
        return false;
    }
    else {
        return true;
    }
}

void normalizacion_pose(Mat& imagen, const std::string& nombre_imagen, const std::vector<Point>& contorno) {
    Rect rectangulo = boundingRect(contorno);
    // rectangle(imagen, rectangulo, Scalar(255));
    // https://stackoverflow.com/a/8110695
    Mat roi = Mat(imagen, rectangulo).clone();

    double angulo = getOrientation(contorno, imagen); // en radianes
    angulo = angulo*180.0/CV_PI; // pasar a grados
    rotar_imagen(roi, angulo); // ángulo en grados

    imagen = roi;

    std::string ruta = std::string("./NormalizadasSinEjes/") + nombre_imagen;
    // imwrite(ruta, imagen);
    imwrite(ruta, roi);
}

void tallar(const RayT& ray, FloatGrid::Accessor& accessor, tools::VolumeRayIntersector<FloatGrid>& inter) {
    double t0 = 0;    
    double t1 = 0;    

    while (inter.march(t0, t1)) {
        double step = 0.01;
        for (double t = t0; t <= t1; t += step) {
            Vec3T voxel = ray.eye() + ray.dir() * t;
            openvdb::Coord coordenada(voxel.x(), voxel.y(), voxel.z());
            accessor.setValueOff(coordenada, 2); // 2 es el background value
        }
    }

    // inter.march(t0, t1);
    // double step = 0.01;
    // for (double t = t0; t <= t1; t += step) {
    //     Vec3T voxel = ray.eye() + ray.dir() * t;
    //     openvdb::Coord coordenada(voxel.x(), voxel.y(), voxel.z());
    //     accessor.setValueOff(coordenada, 2); // 2 es el background value
    // }
    
}

void reconstruccion_por_imagen(int orden, int num_imagenes, Mat& imagen, FloatGrid::Accessor& accessor, tools::VolumeRayIntersector<FloatGrid>& inter) {
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

void reconstruccion(std::vector<Mat>& imagenes) {
    // 1. Construir el cubo
    openvdb::initialize();
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(2.0);
    openvdb::Vec3f centro = openvdb::Vec3f(0,0,0);
    float arista = 200.0f; // arbitrario

    int dim = int(arista / 2);
    openvdb::Coord ijk; // coordenada que servirá para recorrer el bounding box
    // Hacer que estos índices i, j y k referencien a la coordenada que usamos para recorrer.
    int& i = ijk[0];
    int& j = ijk[1];
    int& k = ijk[2];
    auto accessor = grid->getAccessor();
    for (i = centro[0] - dim; i <= centro[0] + dim; i++) {
        for (j = centro[1] - dim; j <= centro[1] + dim; j++) {
            for (k = centro[2] - dim; k <= centro[2] + dim; k++) {
                accessor.setValue(ijk, 0);
            }
        }
    }
    tools::VolumeRayIntersector<FloatGrid> inter(*grid);
    
    debug("Hola");
    // 2. Tallar el cubo
    for (int i = 0; i < imagenes.size(); i++) {
        reconstruccion_por_imagen(i, imagenes.size(), imagenes[i], accessor, inter);
    }
    
    grid->tree().prune();
    // Metadatos
    grid->insertMeta("arista", openvdb::FloatMetadata(arista));
    openvdb::io::File file("salida.vdb");
    openvdb::GridPtrVec grids;
    grids.push_back(grid);
    file.write(grids);
    file.close();
}

void previo(std::vector<Mat>& imagenes) {
    // for (int i = 0; i < imagenes.size(); i++) {
    //     std::cout << "Ancho: " << imagenes[i].cols << "\n";
    //     std::cout << "Altura: " << imagenes[i].rows << "\n";
    // }
    for (int i = 0; i < imagenes.size(); i++) {
        resize(imagenes[i], imagenes[i], Size(200, 200), 0, 0, INTER_CUBIC);
    }

}

int main() {
    std::vector<std::string> nombresImagenes_OE1;
    std::vector<Mat> imagenes_OE1;
    leer_imagenes(nombresImagenes_OE1, imagenes_OE1);
    std::vector<std::vector<Point> > contornos_OE1;

    std::vector<std::string> nombresImagenes_OE2;
    std::vector<Mat> imagenes_OE2;
    std::vector<std::vector<Point> > contornos_OE2;
    
    // OE1: Mejora de contraste y segmentación
    for (size_t i = 0; i < imagenes_OE1.size(); i++) {
        imagenes_OE1[i] = quitar_margen(imagenes_OE1[i], 4);
        binarizacion(imagenes_OE1[i], nombresImagenes_OE1[i]);
        contorno(imagenes_OE1[i], nombresImagenes_OE1[i], contornos_OE1);
        int ind_contorno = 0;
        bool eliminar_imagen = deteccion(imagenes_OE1[i], nombresImagenes_OE1[i], contornos_OE1, ind_contorno);

        if (!eliminar_imagen) {
            imagenes_OE2.push_back(imagenes_OE1[i].clone());
            nombresImagenes_OE2.push_back(nombresImagenes_OE1[i]);
            contornos_OE2.push_back(contornos_OE1[ind_contorno]);
        }

    }

    // OE2: Normalización de pose
    for (size_t i = 0; i < imagenes_OE2.size(); i++) {
        normalizacion_pose(imagenes_OE2[i], nombresImagenes_OE2[i], contornos_OE2[i]);
    }

    // OE3: Reconstrucción
    auto it_ini = imagenes_OE2.begin();
    std::vector<Mat> imagenes_OE3(it_ini, std::next(it_ini, 10)); // coger las 10 primeras imágenes por mientras

    previo(imagenes_OE3);

    reconstruccion(imagenes_OE3);


}
