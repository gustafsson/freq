//g++ -c -o launcher-mac.o launcher-mac.cpp

#include "launcher.h"

#include <CoreFoundation/CFUserNotification.h>
#include <CoreFoundation/CFBundle.h>
#include <unistd.h>
#include <stdio.h>
#include <dlfcn.h>
#include <sys/wait.h>

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

// Application paths for Sonic AWE
char *get_app_path_cuda()
{
    char p[2048];
    char *path = new char[2048];
    sprintf(path, "%s/Contents/MacOS/sonicawe-cuda", bundlePath(p));
    return path;
}
char *get_app_path_cpu()
{
    char p[2048];
    char *path = new char[2048];
    sprintf(path, "%s/Contents/MacOS/sonicawe", bundlePath(p));
    return path;
}
char *get_app_path_opencl()
{
    char p[2048];
    char *path = new char[2048];
    sprintf(path, "%s/Contents/MacOS/sonicawe-opencl", bundlePath(p));
    return path;
}

// Test different environments
int test_cuda_func()
{
    bool test = false;
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
    return 1;
}

// Run the Sonic AWE application
int run(int argc, char *argv[])
{
    int pid, status;
    pid = fork();
    
    if(pid != 0)
    {
        execv(argv[0], argv);
    }
    else
    {
        wait(&status);
    }
    
    return status;
}

// Report errors
void report_error(char *str)
{
    printf("Error: %s\n", str);
}