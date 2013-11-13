#ifndef CUFFTHANDLECONTEXT_H
#define CUFFTHANDLECONTEXT_H

#include "cufft.h"
#include "ThreadChecker.h"

class CufftHandleContext {
public:
    CufftHandleContext( cudaStream_t _stream=0, unsigned type=-1); // type defaults to CUFFT_C2C
    ~CufftHandleContext();

    CufftHandleContext( const CufftHandleContext& b );
    CufftHandleContext& operator=( const CufftHandleContext& b );

    cufftHandle operator()( unsigned elems, unsigned batch_size );

    void setType(unsigned type);

private:
    ThreadChecker _creator_thread;
    cufftHandle _handle;
    cudaStream_t _stream;
    unsigned _type;
    unsigned _elems;
    unsigned _batch_size;

    void destroy();
    void create();
};


#endif // CUFFTHANDLECONTEXT_H
