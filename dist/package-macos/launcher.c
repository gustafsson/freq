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
    char app_path[2048];
    
    // Get the sonicawe application path.
    sprintf(app_path, "%s/Contents/MacOS/sonicawe", bundlePath(path));
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
            CFStringCreateWithCString(NULL, get_error_title(), kCFStringEncodingASCII),
            CFStringCreateWithCString(NULL, get_error_message(), kCFStringEncodingASCII),
            CFStringCreateWithCString(NULL, get_quit(), kCFStringEncodingASCII),
            CFStringCreateWithCString(NULL, get_check_requirements(), kCFStringEncodingASCII),
            CFStringCreateWithCString(NULL, get_get_driver(), kCFStringEncodingASCII),
            &responseFlags);
        
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
            a[0] = (char*)get_browser_bin();
            a[1] = (char*)get_driver_download();
            a[2] = NULL;
    
            printf("Getting CUDA drivers.\n");
            execv(a[0], a);
        }
        else
        {
            // Quit and load the availablity site
            char *a[3];
            a[0] = (char*)get_browser_bin();
            a[1] = (char*)get_requirements_page();
            a[2] = NULL;
    
            printf("Checking requirements.\n");
            execv(a[0], a);
        }
    }
    
    dlclose(test);
    printf("Starting Sonic AWE\n");
    execv(app_path, argv);
    
    return 0;
}