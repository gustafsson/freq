//g++ -c -o launcher.o launcher.c
//g++ -framework CoreFoundation -o launcher launcher.o launcher-mac.o

#include "launcher.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[])
{
    // Set up application paths
    char *app_path_cuda = get_app_path_cuda();
    char *app_path_cpu = get_app_path_cpu();
    char *app_path_opencl = get_app_path_opencl();
    
	char* app_path = app_path_cpu;


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
    printf("Starting %s\n", app_path);
    argv[0] = app_path;
    int return_code = run(argc, argv);

    if (return_code==1337 && strcmp(app_path, app_path_cuda)==0) 
    {
        argv[0] = app_path_cpu;
        run(argc, argv);
    }
    
    return 0;
}