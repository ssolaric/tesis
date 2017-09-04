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


Mat quitar_margen(const Mat& imagen, int margen) {
    Mat nueva(imagen.rows - 2*margen, imagen.cols - 2*margen, imagen.type());
    for (int r = margen; r < imagen.rows - margen; r++) {
        for (int c = margen; c < imagen.cols - margen; c++) {
            nueva.at<Vec3b>(r-margen, c-margen) = imagen.at<Vec3b>(r, c);
        }
    }
    return nueva;
}

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


bool ordenar_por_area(const std::vector<Point>& p1, const std::vector<Point>& p2) {
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

// Retorna true si hay que eliminar la imagen, false en caso contrario.
bool deteccion(Mat& imagen, const std::string& nombre_imagen, std::vector<std::vector<Point> >& contours, int& contorno_cabeza) {
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
    // params.maxInertiaRatio = 0.9;

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
    
        Mat imagenFinal(imagen.size(), CV_8UC1, Scalar(0));
        // Grafico la región del keypoint enocntrado
        drawContours(imagenFinal, contours, contorno_cabeza, Scalar(255), -1);
        std::string ruta = std::string("./Blobs/") + nombre_imagen;
        // imwrite(ruta, imagen_con_keypoints);
        imwrite(ruta, imagenFinal);

        // guardar la imagen creada
        imagen = imagenFinal;
        return false;
    }
    else {
        return true;
    }
}

// void normalizacion_pose(Mat& imagen, const std::string& nombre_imagen, const std::vector<Point>& contorno) {
//     RotatedRect rectangulo = minAreaRect(contorno);
//     Point2f puntos[4];
//     rectangulo.points(puntos);
    
//     Mat imagen_cortada;
//     drawContours(imagen_cortada, contours, contorno_cabeza, Scalar(255), -1);
//     imwrite(contorno);
// }

void drawAxis(Mat& img, Point p, Point q, Scalar colour, const float scale = 0.2)
{
    double angle;
    double hypotenuse;
    angle = atan2( (double) p.y - q.y, (double) p.x - q.x ); // angle in radians
    hypotenuse = sqrt( (double) (p.y - q.y) * (p.y - q.y) + (p.x - q.x) * (p.x - q.x));
//    double degrees = angle * 180 / CV_PI; // convert radians to degrees (0-180 range)
//    cout << "Degrees: " << abs(degrees - 180) << endl; // angle in 0-360 degrees range
    // Here we lengthen the arrow by a factor of scale
    q.x = (int) (p.x - scale * hypotenuse * cos(angle));
    q.y = (int) (p.y - scale * hypotenuse * sin(angle));
    line(img, p, q, colour, 1, CV_AA);
    // create the arrow hooks
    p.x = (int) (q.x + 9 * cos(angle + CV_PI / 4));
    p.y = (int) (q.y + 9 * sin(angle + CV_PI / 4));
    line(img, p, q, colour, 1, CV_AA);
    p.x = (int) (q.x + 9 * cos(angle - CV_PI / 4));
    p.y = (int) (q.y + 9 * sin(angle - CV_PI / 4));
    line(img, p, q, colour, 1, CV_AA);
}
double getOrientation(const std::vector<Point> &pts, Mat &img)
{
    //Construct a buffer used by the pca analysis
    int sz = static_cast<int>(pts.size());
    Mat data_pts = Mat(sz, 2, CV_64FC1);
    for (int i = 0; i < data_pts.rows; ++i)
    {
        data_pts.at<double>(i, 0) = pts[i].x;
        data_pts.at<double>(i, 1) = pts[i].y;
    }
    //Perform PCA analysis
    PCA pca_analysis(data_pts, Mat(), CV_PCA_DATA_AS_ROW);
    //Store the center of the object
    Point cntr = Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)),
                      static_cast<int>(pca_analysis.mean.at<double>(0, 1)));
    //Store the eigenvalues and eigenvectors
    std::vector<Point2d> eigen_vecs(2);
    std::vector<double> eigen_val(2);
    for (int i = 0; i < 2; ++i)
    {
        eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
                                pca_analysis.eigenvectors.at<double>(i, 1));
        eigen_val[i] = pca_analysis.eigenvalues.at<double>(0, i);
    }
    // Draw the principal components
    circle(img, cntr, 3, Scalar(255, 0, 255), 2);
    Point p1 = cntr + 0.02 * Point(static_cast<int>(eigen_vecs[0].x * eigen_val[0]), static_cast<int>(eigen_vecs[0].y * eigen_val[0]));
    Point p2 = cntr - 0.02 * Point(static_cast<int>(eigen_vecs[1].x * eigen_val[1]), static_cast<int>(eigen_vecs[1].y * eigen_val[1]));
    drawAxis(img, cntr, p1, Scalar(0, 255, 0), 1);
    drawAxis(img, cntr, p2, Scalar(255, 255, 0), 5);
    double angle = atan2(eigen_vecs[0].y, eigen_vecs[0].x); // orientation in radians
    return angle;
}

void rotar_imagen(Mat& imagen, double angulo) {
    Point2f centro(imagen.cols/2.0, imagen.rows/2.0);
    Mat rotacion = getRotationMatrix2D(centro, angulo, 1.0); // este ángulo es en grados

    Rect bbox = RotatedRect(centro, imagen.size(), angulo).boundingRect();

    rotacion.at<double>(0, 2) += bbox.width/2.0 - centro.x;
    rotacion.at<double>(1, 2) += bbox.height/2.0 - centro.y;

    warpAffine(imagen, imagen, rotacion, bbox.size());
}

void normalizacion_pose(Mat& imagen, const std::string& nombre_imagen, const std::vector<Point>& contorno) {
    Rect rectangulo = boundingRect(contorno);
    rectangle(imagen, rectangulo, Scalar(255));

    // https://stackoverflow.com/a/8110695
    Mat roi = Mat(imagen, rectangulo).clone();
    
    // double angulo = getOrientation(contorno, roi);
    double angulo = getOrientation(contorno, imagen); // en radianes
    // angulo = std::min(angulo, CV_PI -angulo);
    debug(angulo*180.0/CV_PI);
    
    // rotar_imagen(roi, -angulo);
    rotar_imagen(imagen, angulo*180.0/CV_PI); // ángulo en grados

    std::string ruta = std::string("./Normalizadas/") + nombre_imagen;
    // imwrite(ruta, roi);
    imwrite(ruta, imagen);
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

}

// TODO: preguntar cómo hacer la segmentación de las cabezas y preparar la normalización de pose