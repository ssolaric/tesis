#pragma once

#include <openvdb/openvdb.h>
#include <openvdb/Types.h>
#include <openvdb/tools/RayIntersector.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/VolumeToMesh.h>

#define debug(x) do { std::cout << #x << ": " << x << "\n"; } while (0)

typedef openvdb::math::Ray<double> RayT;
typedef RayT::Vec3Type Vec3T;

const int BACKGROUND_VALUE = 1;

void tallar(const RayT& ray, openvdb::FloatGrid::Accessor& accessor, openvdb::tools::VolumeRayIntersector<openvdb::FloatGrid>& inter, double step) {
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
    for (double t = t0; t <= t1; t += step) {
        Vec3T voxel = ray.eye() + ray.dir() * t;
        openvdb::Coord coordenada(voxel.x(), voxel.y(), voxel.z());
        accessor.setValueOff(coordenada, BACKGROUND_VALUE);
    }
    
}

void reconstruccionPorImagen(int orden, int num_imagenes, cv::Mat& imagen, openvdb::FloatGrid::Accessor& accessor, openvdb::tools::VolumeRayIntersector<openvdb::FloatGrid>& inter, double step) {
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
                    tallar(ray, accessor, inter, step);
                }
            }
        }
    }
}

// http://www.openvdb.org/documentation/doxygen/namespaceopenvdb_1_1v4__0__2_1_1tools.html#a1cc2d4f60d561afd2fb248e386ca67d8

template<typename GridOrTreeType>
typename GridOrTreeType::Ptr limpiarEscena(GridOrTreeType& volume) {
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

void guardarMalla(const std::vector<openvdb::Vec3s>& puntos, const std::vector<openvdb::Vec4I>& cuadrilateros, const std::string& nombreMalla) {
    std::ofstream archivo(nombreMalla + ".off");
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
void reconstruccion(std::vector<cv::Mat>& imagenes, double step, const std::string& nombreMalla) {
    // 1. Construir el cubo
    openvdb::initialize();
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(BACKGROUND_VALUE); // 1 es el background value
    openvdb::CoordBBox caja(openvdb::Coord(-100, -100, -100), openvdb::Coord(100, 100, 100));
    grid->fill(caja, -1); // -1 es el valor de un voxel activo (ya no se mostrará en el vdb_view porque este necesita valor 0 para mostrar)

    auto accessor = grid->getAccessor();
    openvdb::tools::VolumeRayIntersector<openvdb::FloatGrid> inter(*grid);    
    // 2. Tallar el cubo
    for (int i = 0; i < imagenes.size(); i++) {
        reconstruccionPorImagen(i, imagenes.size(), imagenes[i], accessor, inter, step);
    }

    // 3. Limpiar la escena
    openvdb::FloatGrid::Ptr espermatozoide = limpiarEscena(*grid);

    // 4. Pasarla a malla.
    // Así ya no tendría que usar el visor del openvdb.
    std::vector<openvdb::Vec3s> puntos;
    std::vector<openvdb::Vec4I> cuadrilateros;
    openvdb::tools::volumeToMesh(*espermatozoide, puntos, cuadrilateros);
    debug(puntos.size());
    debug(cuadrilateros.size());
    guardarMalla(puntos, cuadrilateros, nombreMalla);
}