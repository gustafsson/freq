#include "cuffthandlecontext.h"

#include "TaskTimer.h"
#include "CudaException.h"

CufftHandleContext::
        CufftHandleContext( cudaStream_t stream, unsigned type )
:   _handle(0),
    _stream(stream),
    _type(type)
{
    if (_type == (unsigned)-1)
        _type = CUFFT_C2C;
}


CufftHandleContext::
        ~CufftHandleContext()
{
    destroy();
    _creator_thread.reset();
}


CufftHandleContext::
        CufftHandleContext( const CufftHandleContext& b )
{
    _handle = 0;
    this->_stream = b._stream;
    this->_type = b._type;
}


CufftHandleContext& CufftHandleContext::
        operator=( const CufftHandleContext& b )
{
    destroy();
    this->_stream = b._stream;
    this->_type = b._type;
    return *this;
}


cufftHandle CufftHandleContext::
        operator()( unsigned elems, unsigned batch_size )
{
    if (0 == _handle || _elems != elems || _batch_size != batch_size) {
        this->_elems = elems;
        this->_batch_size = batch_size;
        create();
    } else {
        _creator_thread.throwIfNotSame(__FUNCTION__);
    }
    return _handle;
}


void CufftHandleContext::
        setType(unsigned type)
{
    if (this->_type != type)
    {
        destroy();
        this->_type = type;
    }
}


void CufftHandleContext::
        create()
{
    destroy();

    if (_elems == 0 || _batch_size==0)
        return;

    int n = _elems;
    cufftResult r = cufftPlanMany(
            &_handle,
            1,
            &n,
            NULL, 1, 0,
            NULL, 1, 0,
            (cufftType_t)_type,
            _batch_size);

    if (CUFFT_SUCCESS != r)
    {
        TaskInfo ti("cufftPlanMany( n = %d, _batch_size = %u ) -> %s",
                    n, _batch_size, CufftException::getErrorString(r));
        size_t free=0, total=0;
        cudaMemGetInfo(&free, &total);
        ti.tt().info("Free mem = %g MB, total = %g MB", free/1024.f/1024.f, total/1024.f/1024.f);
        CufftException_SAFE_CALL( r );
    }

    CufftException_SAFE_CALL(cufftSetStream(_handle, _stream ));
    _creator_thread.reset();
}


void CufftHandleContext::
        destroy()
{
    if (_handle!=0) {
        _creator_thread.throwIfNotSame(__FUNCTION__);

        if (_handle==(cufftHandle)-1)
            TaskInfo("CufftHandleContext::destroy, _handle==(cufftHandle)-1");
        else
        {
            cufftResult errorCode = cufftDestroy(_handle);
            if (errorCode != CUFFT_SUCCESS)
                TaskInfo("CufftHandleContext::destroy, %s", CufftException::getErrorString(errorCode) );
        }

        _handle = 0;
    }
}
