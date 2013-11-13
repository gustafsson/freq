//g++ -c -o launcher-mac.o launcher-mac.cpp

#include "launcher.h"

#include <CoreFoundation/CFUserNotification.h>
#include <CoreFoundation/CFBundle.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <sys/wait.h>

using namespace std;

// Windows specific code!


// Mac OS specific code!


// Gets application bundle path
const char *bundlePath(char *path)
{
    CFBundleRef mainBundle = CFBundleGetMainBundle();
     
    CFURLRef mainBundleURL = CFBundleCopyBundleURL(mainBundle);
     
    CFStringRef cfStringRef = CFURLCopyFileSystemPath(mainBundleURL, kCFURLPOSIXPathStyle);
     
    CFStringGetCString(cfStringRef, path, 1024, kCFStringEncodingASCII);
     
    CFRelease(mainBundleURL);
    CFRelease(cfStringRef);
     
    return path;
}

string get_mac_origin()
{
    char p[2048];
    string s = bundlePath(p);
    s += "/Contents/MacOS/sonicawe";
    return s;
}

// Application paths for Sonic AWE
string get_app_path_cuda()
{
    string s = get_mac_origin() + "-cuda";
    return s;
}
string get_app_path_cpu()
{
    string s = get_mac_origin();
    return s;
}
string get_app_path_opencl()
{
    string s = get_mac_origin() + "-opencl";
    return s;
}

// Test different environments
int test_cuda_func()
{
    bool test = 0;
    // Try to load the CUDA library (Checking for CUDA enabled drivers)
    void* nvcuda = dlopen("/usr/local/cuda/lib/libcuda.dylib", RTLD_LAZY);
    
    if (nvcuda) {
        typedef int (*cuInitFunction)(int*);
        
        cuInitFunction cuInit = (cuInitFunction)dlsym(nvcuda, "cuInit"); 
        if (cuInit && cuInit(0)==0) 
            test = 1;
        
        dlclose(nvcuda);
    }
    return test;
}


int test_opencl_func()
{
    return 0;
}

// Run the Sonic AWE application
int run(int argc, char *argv[])
{
    const int launch_failed = -17;
    
    if(0 == fork())
    {
        if (execv(argv[0], argv) < 0)
            exit(launch_failed);
    }

    int status;
    wait(&status);
    if (WIFEXITED(status))
        return (char)WEXITSTATUS(status);
    else
        return -1;
}

// Report errors
void report_error(char *str)
{
    printf("Error: %s\n", str);
}