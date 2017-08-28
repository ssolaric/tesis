#include <dirent.h>
#include <stdlib.h>
#include <stdio.h>

#include <regex>
#include <iostream>

int main() {
    struct dirent** namelist;
    int n = scandir("./Espermatozoides", &namelist, NULL, versionsort);

    std::regex numeros_jpg("^\\d+\\.jpg");
    std::regex out_numeros_jpg("^out_img\\d+\\.jpg");
    std::vector<std::string> cadenas;

    int cont = 0;
    
    if (n < 0) {
        perror("scandir");
    }
    else {
        // parseamos los de la forma 12.jpg
        for (int i = 0; i < n; i++) {
            if (std::regex_search(namelist[i]->d_name, numeros_jpg)) {
                cont++;
                std::string ruta_actual = std::string("Espermatozoides/") + namelist[i]->d_name;
                std::string ruta_nueva = std::string("Imagenes/") + std::to_string(cont) + ".jpg";
                rename(ruta_actual.c_str(), ruta_nueva.c_str());
                cadenas.push_back(std::to_string(cont) + ".jpg");
            }
        }

        // parseamos los de la forma out_img270.jpg
        for (int i = 0; i < n; i++) {
            if (std::regex_search(namelist[i]->d_name, out_numeros_jpg)) {
                cont++;
                std::string ruta_actual = std::string("Espermatozoides/") + namelist[i]->d_name;
                std::string ruta_nueva = std::string("Imagenes/") + std::to_string(cont) + ".jpg";
                rename(ruta_actual.c_str(), ruta_nueva.c_str());
                cadenas.push_back(std::to_string(cont) + ".jpg");
            }
            free(namelist[i]);
        }
        free(namelist);
    }

    for (auto s : cadenas) {
        std::cout << s << "\n";
    }


    return 0;
}
