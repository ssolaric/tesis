#define debug(x) do { std::cout << #x << ": " << x << "\n"; } while (0)
#include <openvdb/openvdb.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <vector>
#include <fstream>
using namespace openvdb;

void guardar_malla(const std::vector<openvdb::Vec3s>& puntos, const std::vector<openvdb::Vec3I>& triangulos) {
    std::ofstream archivo("malla.off");
    archivo << "OFF\n";
    archivo << puntos.size() << " " << triangulos.size() << " 0\n";
    for (int i = 0; i < puntos.size(); i++) {
        archivo << puntos[i].x() << " " << puntos[i].y() << " " << puntos[i].z() << "\n";
    }

    for (int i = 0; i < triangulos.size(); i++) {
        archivo << "3 " << triangulos[i].x() << " " << triangulos[i].y() << " " << triangulos[i].z() << "\n";
    }
}

int main() {
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(1.0);
    CoordBBox caja(Coord(-100, -100, -100), Coord(100, 100, 100));
    grid->fill(caja, -1);
    
    // std::vector<openvdb::Vec3s> points;
    // std::vector<openvdb::Vec4I> quads;
    // std::vector<openvdb::Vec3I> triangles;

    // openvdb::tools::volumeToMesh(*grid, points, quads);
    // typedef openvdb::tree::Tree4<float, 5, 4, 3>::Type  FloatTreeType;
    // typedef openvdb::Grid<FloatTreeType>                FloatGridType;

    // FloatGridType grid(1.0f);

    // // test voxel region meshing

    // openvdb::CoordBBox bbox(openvdb::Coord(1), openvdb::Coord(6));

    // grid.tree().fill(bbox, -1.0f);

    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec4I> quads;
    std::vector<openvdb::Vec3I> triangles;

    openvdb::tools::volumeToMesh(*grid, points, triangles, quads);
    guardar_malla(points, triangles);
    debug(points.size());
    debug(triangles.size());
    debug(quads.size());
    
}