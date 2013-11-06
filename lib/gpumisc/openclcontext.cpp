#include "openclcontext.h"
#include "TaskTimer.h"
#include <stdexcept>
#include <QMutex>

void error_notify(
        const char *errinfo,
        const void */*private_info*/, size_t /*cb*/,
        void *user_data)
{
    ((OpenCLContext*)user_data)->error( errinfo );
}


OpenCLContext::OpenCLContext(cl_device_type deviceType)
    :
    callback(0)
{
    cl_int err;
    cl_device_id device_ids[16];
    cl_platform_id platforms[16];
    unsigned int num_devices;

#ifndef __APPLE__
    err = clGetPlatformIDs(16, platforms, NULL);
    if(err)
    {
        throw std::runtime_error("Couldn't get OpenCL platform");
    }

    err = clGetDeviceIDs(platforms[0], deviceType, 16, device_ids, &num_devices);
#else
    err = clGetDeviceIDs(NULL, deviceType, 16, device_ids, &num_devices);
#endif

    if(err)
    {
        throw std::runtime_error("Couldn't get OpenCL device");
    }

    deviceId_ = NULL;

    unsigned int i;
    for(i = 0; i < num_devices; i++)
    {
        cl_bool available;
        err = clGetDeviceInfo(device_ids[i], CL_DEVICE_AVAILABLE, sizeof(cl_bool), &available, NULL);
        char name[200]="";
        err |= clGetDeviceInfo(device_ids[i], CL_DEVICE_NAME, sizeof(name), name, NULL);

        if(err)
        {
            TaskInfo("Cannot check availability of device # %d", i);
        }

        if(available)
        {
            // TODO: Choose best GPGPU or CPU for the job!
            deviceId_ = device_ids[i];
            deviceName_ = name;
            TaskInfo("Selecting device %s for OpenCL context", name);
            break;
        }
        else
        {
            if(err == CL_SUCCESS)
            {
                TaskInfo("Device %s not available for OpenCL", name);
            }
            else
            {
                TaskInfo("Device # %d not available for OpenCL", i);
            }
        }
    }

    if(!deviceId_)
    {
        throw std::runtime_error("No available devices found for OpenCL.");
    }

    context_ = clCreateContext(0, 1, &deviceId_, error_notify, this, &err);
    if(!context_ || err)
    {
        throw std::runtime_error("clCreateContext failed");
    }

    queue_ = clCreateCommandQueue(context_, deviceId_, 0, &err);
    if(!queue_ || err)
    {
        clReleaseContext(context_);
        throw std::runtime_error("clCreateCommandQueue() failed.");
    }
}

OpenCLContext::~OpenCLContext()
{
    clReleaseContext(context_);
    clReleaseCommandQueue(queue_);
}


void OpenCLContext::
        error(const char* errinfo)
{
    TaskInfo("OpenCL error: %s", errinfo);

    if (callback)
        callback->error( errinfo );
}


void OpenCLContext::
        registerCallback( OpenCLContextCallback* callback )
{
    this->callback = callback;
}


cl_ulong OpenCLContext::getMemorySize()
{
    cl_int err;
    cl_ulong gMemSize;
    err = clGetDeviceInfo(deviceId_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &gMemSize, NULL);
    if(err)
    {
        throw std::runtime_error("Failed to get global mem size");
    }
    return gMemSize;
}


std::string OpenCLContext::
        deviceName()
{
    return deviceName_;
}


cl_context OpenCLContext::getContext()
{
    return context_;
}
cl_command_queue OpenCLContext::getCommandQueue()
{
    return queue_;
}
cl_device_id OpenCLContext::getDeviceID()
{
    return deviceId_;
}
