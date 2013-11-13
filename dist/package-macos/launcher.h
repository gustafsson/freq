#include <string>

// Application paths for Sonic AWE
std::string get_app_path_cuda();
std::string get_app_path_cpu();
std::string get_app_path_opencl();

// Test different environments
int test_cuda_func();
int test_opencl_func();

// Run the Sonic AWE application
int run(int argc, char *argv[]);

// Error reporting function
void report_error(char *str);