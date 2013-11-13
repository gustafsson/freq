#ifndef OPENCLCONTEXT_H
#define OPENCLCONTEXT_H
#ifdef __APPLE__
#	include <OpenCL/opencl.h>
#else
#	include <CL/opencl.h>
#endif
#include "HasSingleton.h"

#include <string>

class OpenCLContextCallback
{
public:
    virtual void error(const char* errinfo) = 0;
};

class OpenCLContext: public HasSingleton<OpenCLContext>, OpenCLContextCallback
{
public:
    ~OpenCLContext();

    cl_context getContext();
    cl_command_queue getCommandQueue();
    cl_device_id getDeviceID();
    cl_ulong getMemorySize();
    std::string deviceName();
    void error(const char* errinfo);

    /**
      If you register a callback you must take care of thread safety.
      OpenCLContextCallback::error will be called from an OpenCL driver thread.
      */
    void registerCallback( OpenCLContextCallback* callback );

protected:
    cl_device_id deviceId_;
    std::string deviceName_;
    cl_context context_;
    cl_command_queue queue_;
    OpenCLContextCallback* callback;

private:
    friend class HasSingleton<OpenCLContext>;
    OpenCLContext(cl_device_type deviceType = CL_DEVICE_TYPE_GPU); // CL_DEVICE_TYPE_GPU is required for clFFT!
};

#endif // OPENCLCONTEXT_H
