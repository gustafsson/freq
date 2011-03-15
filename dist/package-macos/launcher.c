//gcc -framework CoreFoundation -o launcher launcher.c

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
        const char* title = "Couldn't find CUDA, cannot start Sonic AWE";
        const char* msg = 
			"Sonic AWE requires you to have installed recent display drivers from NVIDIA, and no such driver was found.\n"
			"\n"
            "Hardware requirements: You need to have one of these graphics cards from NVIDIA:\n"
            "   www.nvidia.com/object/cuda_gpus.html\n"
            "\n"
            "\nSoftware requirements: You also need to have installed recent display drivers from NVIDIA:\n"
            "\n"
            "\"Developer Drivers for MacOS\" at \nhttp://www.nvidia.com/object/cuda_get.html#MacOS\n"
            "\n"
            "\nSonic AWE cannot start. Please try again with updated drivers.";

        CFOptionFlags options = kCFUserNotificationStopAlertLevel | kCFUserNotificationNoDefaultButtonFlag;
        CFOptionFlags responseFlags = 0;
        CFUserNotificationDisplayAlert(0, options, NULL, NULL, NULL,
            CFSTR(title),
            CFSTR(msg),
            CFSTR("Ok"), NULL, NULL, &responseFlags);
        return 1;
    }
    else
    {
        fclose(pFile);
        execv(app_path, argv);
    }
}
