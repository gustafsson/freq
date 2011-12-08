#include <windows.h>
#include <iostream>
#include <process.h>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

using namespace std;

int main(int argc, char *argv[])
{
    char app_path_cuda[] = TOSTRING(PACKAGE) "-cuda.exe";
    char app_path_cpu[] = TOSTRING(PACKAGE) "-cpu.exe";
    char* app_path = app_path_cpu;

    HMODULE nvcuda = LoadLibrary(L"nvcuda.dll");

    if (nvcuda) {
        typedef int (WINAPI *cuInitFunction)(int*);
        
        cuInitFunction cuInit = (cuInitFunction)GetProcAddress(nvcuda, "cuInit"); 
        if (cuInit && cuInit(0)==0) 
            app_path = app_path_cuda;
        
        FreeLibrary(nvcuda);
    }

    printf("Starting %s\n", app_path);
    argv[0] = app_path;
    int return_code = system(app_path);

    if (return_code==1337 && strcmp(app_path, app_path_cuda)==0) 
        system(app_path_cpu);

    argc; // suppress warning

    return 0;
}
