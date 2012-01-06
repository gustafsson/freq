//gcc -framework CoreFoundation -o launcher launcher.c

#include <CoreFoundation/CFUserNotification.h>
#include <CoreFoundation/CFBundle.h>
#include <unistd.h>
#include <stdio.h>
#include <dlfcn.h>

#include "common_message.h"

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

int main(int argc, char *argv[])
{
    char path[2048];
    char app_path_cuda[2048];
    char app_path_cpu[2048];
    
    // Get the sonicawe application path.
    sprintf(app_path_cuda, "%s/Contents/MacOS/sonicawe-cuda", bundlePath(path));
    sprintf(app_path_cpu, "%s/Contents/MacOS/sonicawe", bundlePath(path));
    
	char* app_path = app_path_cpu;

    // Option flags for notification
    CFOptionFlags options = kCFUserNotificationStopAlertLevel | kCFUserNotificationNoDefaultButtonFlag;
    
    // The response from the notification
    CFOptionFlags responseFlags = 0;
    
    // Try to load the CUDA library (Checking for CUDA enabled drivers)
    void* nvcuda = dlopen("/usr/local/cuda/lib/libcuda.dylib", RTLD_LAZY);
    
    if (nvcuda) {
        typedef int (*cuInitFunction)(int*);
        
        cuInitFunction cuInit = (cuInitFunction)dlsym(nvcuda, "cuInit"); 
        if (cuInit && cuInit(0)==0) 
            app_path = app_path_cuda;
        
        dlclose(nvcuda);
    }

    printf("Starting %s\n", app_path);
    argv[0] = app_path;
	execv(app_path, argv);
    
    return 0;
}