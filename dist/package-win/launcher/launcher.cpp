#include <windows.h>
#include <iostream>
#include <process.h>
#include <sstream>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

using namespace std;

int run(int argc, char *argv[])
{
    STARTUPINFO si;
    PROCESS_INFORMATION pi;

    ZeroMemory( &si, sizeof(si) );
    si.cb = sizeof(si);
    ZeroMemory( &pi, sizeof(pi) );

    std::wstringstream commandline;

    for (int i=0; i<argc; ++i)
        commandline << argv[i] << " ";

    // Start the child process.
    if( !CreateProcess( NULL,   // No module name (use command line)
        (LPWSTR)commandline.str().c_str(),        // Command line
        NULL,           // Process handle not inheritable
        NULL,           // Thread handle not inheritable
        FALSE,          // Set handle inheritance to FALSE
        0,              // No creation flags
        NULL,           // Use parent's environment block
        NULL,           // Use parent's starting directory
        &si,            // Pointer to STARTUPINFO structure
        &pi )           // Pointer to PROCESS_INFORMATION structure
    )
    {
        OutputDebugString( L"CreateProcess failed\n" );
        return -17;
    }

    // Wait until child process exits.
    WaitForSingleObject( pi.hProcess, INFINITE );

    // Get exit code
    DWORD exitCode = -1;
    GetExitCodeProcess( pi.hProcess, &exitCode );

    // Close process and thread handles.
    CloseHandle( pi.hProcess );
    CloseHandle( pi.hThread );

    return (char)exitCode;
}

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
    int return_code = run(argc, argv);

    if (return_code == -1 || return_code==-17 && strcmp(app_path, app_path_cuda)==0)
    {
        argv[0] = app_path_cpu;
        run(argc, argv);
    }

    return 0;
}
