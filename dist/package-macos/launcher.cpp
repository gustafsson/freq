//g++ -c -o launcher.o launcher.c
//g++ -framework CoreFoundation -o launcher launcher.o launcher-mac.o

#include "launcher.h"
#include <stdio.h>
#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{
    // Set up application paths
    string app_path_cuda = get_app_path_cuda();
    string app_path_cpu = get_app_path_cpu();
    string app_path_opencl = get_app_path_opencl();
    
	string app_path = app_path_cpu;


    // Checking which platform to load
    if(test_cuda_func())
    {
        app_path = app_path_cuda;
    }
    else if(test_opencl_func())
    {
        app_path = app_path_opencl;
    }
    else
    {
        app_path = app_path_cpu;
    }

    // Running the application
    cout<<"Starting "<<app_path.c_str()<<"\n";
    argv[0] = (char*)(app_path.c_str());
    int return_code = run(argc, argv);

    if (return_code == -17 && app_path.compare(app_path_cuda) == 0)
    {
        argv[0] = (char*)(app_path_cpu.c_str());
        int a = run(argc, argv);
    }
    
    return 0;
}