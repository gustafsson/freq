#include <windows.h>
#include <stdio.h>
#include <process.h>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

int main(int argc, char *argv[])
{
    char app_path_cuda[] = TOSTRING(PACKAGE) "-cuda.exe";
    char app_path_cpu[] = TOSTRING(PACKAGE) "-cpu.exe";
    char* app_path = app_path_cuda;
    // Try to load the CUDA library (Checking for CUDA enabled drivers)
	HMODULE test = LoadLibrary(L"nvcuda.dll");
    
    if ( test == NULL ) 
    {
        app_path = app_path_cpu;
    }
    else
    {
        FreeLibrary(test);
    }

    printf("Starting %s\n", app_path);
    argv[0] = app_path;
    _execv(app_path, argv);

    argc; // suppress warning

    return 0;
}
