#include "openclmemorystorage.h"

#include "cpumemorystorage.h"
#include "openclcontext.h"
#include "TaskTimer.h"

OpenClMemoryStorage::
        OpenClMemoryStorage( DataStorageVoid* p, bool read, bool write )
            :
            DataStorageImplementation( p ),
            data( 0 ),
            flags( 0 ),
            borrowsData( false )
{
    OpenClMemoryStorage* q = p->FindStorage<OpenClMemoryStorage>();
    EXCEPTION_ASSERT( q == this );

    if (!read)
        flags = CL_MEM_WRITE_ONLY;
    else if (!write)
        flags = CL_MEM_READ_ONLY;
    else
        flags = CL_MEM_READ_WRITE;
}


OpenClMemoryStorage::
        OpenClMemoryStorage( DataStorageVoid* p, cl_mem data, cl_mem_flags flags, bool adoptData )
            :
            DataStorageImplementation( p ),
            data( data ),
            flags( flags ),
            borrowsData( !adoptData )
{
    OpenClMemoryStorage* q = p->FindStorage<OpenClMemoryStorage>();
    EXCEPTION_ASSERT( q == this );

    p->FindCreateStorage<OpenClMemoryStorage>( false, true ); // Mark memory as up to date
}


OpenClMemoryStorage::
        ~OpenClMemoryStorage()
{
    if (!borrowsData)
        clReleaseMemObject(data);
}


/*static*/ OpenClMemoryStorage* OpenClMemoryStorage::
        AccessStorage( DataStorageVoid* dsv, bool read, bool write )
{
    cl_mem_flags incompatible_flags = 0;
    if (!write)
        incompatible_flags = CL_MEM_WRITE_ONLY;
    else if (!read)
        incompatible_flags = CL_MEM_READ_ONLY;

    OpenClMemoryStorage* clmem = dsv->FindStorage<OpenClMemoryStorage>();
    if (clmem && 0!=(clmem->flags & incompatible_flags))
        throw std::runtime_error("An OpenCL kernel can't read from memory created write only (or write to read only).");

    // ok flags, make sure it's up-to-date, or create it if needed
    return dsv->FindCreateStorage<OpenClMemoryStorage>( read, write );
}


bool OpenClMemoryStorage::
        updateFromOther(DataStorageImplementation *p)
{
    OpenCLContext *opencl = &OpenCLContext::Singleton();
    cl_context context = opencl->getContext();
    cl_command_queue queue = opencl->getCommandQueue();
    cl_int fft_error = 0;

    if (CpuMemoryStorage* cpu = dynamic_cast<CpuMemoryStorage*>(p))
    {
        CpuMemoryAccess<char, 3> cpuReader = cpu->AccessBytes<3>();

        if (0!=data)
            fft_error |= clEnqueueWriteBuffer(queue, data, CL_TRUE, 0, dataStorage()->numberOfBytes(), cpuReader.ptr(), 0, NULL, NULL);
        else
            data = clCreateBuffer(context, flags | CL_MEM_COPY_HOST_PTR, dataStorage()->numberOfBytes(), cpuReader.ptr(), &fft_error);

        if(fft_error != CL_SUCCESS)
            throw std::runtime_error("Could not write to OpenCL memory");

        return true;
    }

    if (0!=data) if (OpenClMemoryStorage* b = dynamic_cast<OpenClMemoryStorage*>(p))
    {
        fft_error |= clEnqueueCopyBuffer(queue,
                            b->data,
                            data,
                            0, 0, dataStorage()->numberOfBytes(),
                            0, NULL, NULL);

        if(fft_error != CL_SUCCESS)
            throw std::runtime_error("Could not copy OpenCL memory");

        return true;
    }

    return false;
}


bool OpenClMemoryStorage::
        updateOther(DataStorageImplementation *p)
{
    OpenCLContext *opencl = &OpenCLContext::Singleton();
    cl_command_queue queue = opencl->getCommandQueue();
    cl_int fft_error = 0;

    if (0!=data) if (CpuMemoryStorage* cpu = dynamic_cast<CpuMemoryStorage*>(p))
    {
        CpuMemoryAccess<char, 3> cpuWriter = cpu->AccessBytes<3>();

        fft_error |= clEnqueueReadBuffer(queue, data, CL_TRUE, 0, dataStorage()->numberOfBytes(), cpuWriter.ptr(), 0, NULL, NULL);

        if(fft_error != CL_SUCCESS)
            throw std::runtime_error("Could not read from OpenCL memory");

        return true;
    }

    if (OpenClMemoryStorage* b = dynamic_cast<OpenClMemoryStorage*>(p))
    {
        return b->updateFromOther( this );
    }

    return false;
}


void OpenClMemoryStorage::
        clear()
{
    TaskTimer tt("OpenClMemoryStorage::clear (%u, %u, %u) %s",
                 size().width, size().height, size().depth,
                 DataStorageVoid::getMemorySizeText( dataStorage()->numberOfBytes() ).c_str());

    CpuMemoryStorage* c = dataStorage()->FindCreateStorage<CpuMemoryStorage>( false, true );
    // CpuMemoryStorage is here the only updated storage so calling clear here will not be recursive
    dataStorage()->ClearContents();
    updateFromOther( c );
}


DataStorageImplementation* OpenClMemoryStorage::
        newInstance( DataStorageVoid* p )
{
    return new OpenClMemoryStorage( p, !(flags & CL_MEM_WRITE_ONLY), !(flags & CL_MEM_READ_ONLY));
}


bool OpenClMemoryStorage::
        allowCow()
{
    return !borrowsData;
}
