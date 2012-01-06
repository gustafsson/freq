// Application paths for Sonic AWE
char *get_app_path_cuda();
char *get_app_path_cpu();
char *get_app_path_opencl();

// Test different environments
int test_cuda_func();
int test_opencl_func();

// Run the Sonic AWE application
int run(int argc, char *argv[]);

// Error reporting function
void report_error(char *str);