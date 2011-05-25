// Interwebz linkz
const char *get_driver_download(){return "http://www.nvidia.com/object/cuda_get.html#MacOS";}
const char *get_requirements_page(){return "http://www.nvidia.com/object/cuda_gpus.html";}

// String messages
const char *get_error_title(){return "Couldn't find CUDA, unable to start Sonic AWE";}
const char *get_error_message()
{
    return "Sonic AWE requires you to have a CUDA enabled display driver from NVIDIA, and no such driver was found.\n\nHardware requirements: You need to have one of these graphics cards from NVIDIA: www.nvidia.com/object/cuda_gpus.html\n\nSoftware requirements: You also need to have installed recent display drivers from NVIDIA: (Developer Drivers for MacOS) www.nvidia.com/object/cuda_get.html#MacOS\n\nSonic AWE cannot start. Please try again with updated drivers.";
}
const char *get_quit(){return "Quit";}
const char *get_check_requirements(){return "Check requirements";}
const char *get_get_driver(){return "Get driver";}

// Browser binary
#ifdef __APPLE__
const char *get_browser_bin(){return "/usr/bin/open";}
#else
const char *get_browser_bin(){return "";}
#endif
