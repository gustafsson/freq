//gcc -framework CoreFoundation -o launcher lassare.c

#include <CoreFoundation/CFUserNotification.h>
#include <CoreFoundation/CFBundle.h>
#include <unistd.h>
#include <stdio.h>

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
    FILE * pFile = fopen ("/usr/local/cuda/lib/libcuda.dylib","r");
    char path[2048];
    char app_path[2048];
    sprintf(app_path, "%s/Contents/MacOS/sonicawe", bundlePath(path));
    printf("%s\n", app_path);

    if (pFile==NULL)
    {
        CFOptionFlags options = kCFUserNotificationStopAlertLevel | kCFUserNotificationNoDefaultButtonFlag;
        CFOptionFlags responseFlags = 0;
        CFUserNotificationDisplayAlert(0, options, NULL, NULL, NULL,
            CFSTR("CUDA is not installed!"),
            CFSTR("CUDA could not be found on your computer. You need CUDA in order to run SonicAWE. Please get CUDA!"),
            CFSTR("Ok"), NULL, NULL, &responseFlags);
        return 1;
    }
    else
    {
        fclose(pFile);
        execv(app_path, argv);
    }
}