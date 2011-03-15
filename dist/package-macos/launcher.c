//gcc -framework CoreFoundation -o launcher launcher.c

#include <CoreFoundation/CFUserNotification.h>
#include <CoreFoundation/CFBundle.h>
#include <unistd.h>
#include <stdio.h>
#include <dlfcn.h>

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
    char app_path[2048];
    
    // Get the sonicawe application path.
    sprintf(app_path, "%s/Contents/MacOS/sonicawe_app", bundlePath(path));
    printf("%s\n", app_path);

    // Option flags for notification
    CFOptionFlags options = kCFUserNotificationStopAlertLevel | kCFUserNotificationNoDefaultButtonFlag;
    
    // The response from the notification
    CFOptionFlags responseFlags = 0;
    
    // Try to load the CUDA library (Checking for CUDA enabled drivers)
    void* test = dlopen("/usr/local/cuda/lib/libcuda.dylib", RTLD_LAZY);
    
    if ( test == NULL )
    {
        // Notify the user that CUDA drivers could not be found.
        CFUserNotificationDisplayAlert(0, options, NULL, NULL, NULL,
            CFSTR("Couldn't find CUDA, unable to start Sonic AWE"),
            CFSTR("Sonic AWE requires you to have a CUDA enabled display driver from NVIDIA, and no such driver was found.\n\nHardware requirements: You need to have one of these graphics cards from NVIDIA: www.nvidia.com/object/cuda_gpus.html\n\nSoftware requirements: You also need to have installed recent display drivers from NVIDIA: (Developer Drivers for MacOS) www.nvidia.com/object/cuda_get.html#MacOS\n\nSonic AWE cannot start. Please try again with updated drivers."),
            CFSTR("Quit"), CFSTR("Check requirements"), CFSTR("Get driver"), &responseFlags);
        
        if( (unsigned long)responseFlags == 0)
        {
            // Quit and do nothing
            printf("Quitting.\n");
            return 1;
        }
        else if( (unsigned long)responseFlags == 2)
        {
            // Quit and load the driver download site
            char *a[3];
            a[0] = "/usr/bin/open";
            a[1] = "http://www.nvidia.com/object/cuda_get.html#MacOS";
            a[2] = NULL;
    
            printf("Getting CUDA drivers.\n");
            execv("/usr/bin/open", a);
        }
        else
        {
            // Quit and load the availablity site
            char *a[3];
            a[0] = "/usr/bin/open";
            a[1] = "http://www.nvidia.com/object/cuda_gpus.html";
            a[2] = NULL;
    
            printf("Checking requirements.\n");
            execv("/usr/bin/open", a);
        }
    }
    
    dlclose(test);
    printf("Starting Sonic AWE\n");
    execv(app_path, argv);
    
    return 0;
}