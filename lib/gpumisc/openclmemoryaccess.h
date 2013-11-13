#ifndef OPENCLMEMORYACCESS_H
#define OPENCLMEMORYACCESS_H

#include "CL/cl.h"
#include "datastorageaccess.h"

template<typename DataType, unsigned Dimension>
class OpenClMemoryAccess
{
    // implementes DataStorageAccessConcept

    cl_mem ptr_;
    DataAccessSize<Dimension> sz; // numberOfElements


public:
    typedef DataType T;
    typedef DataAccessPosition<Dimension> Position;
    typedef DataAccessSize<Dimension> Size;


    OpenClMemoryAccess(cl_mem p, DataAccessSize<Dimension> numberOfElements)
        : ptr_(p),
          sz( numberOfElements )
    {
    }


    DataAccessSize<Dimension> numberOfElements() { return sz; }


    T& ref( Position p )
    {
        return ptr_[ sz.offset(p) ];
    }


    cl_mem ptr()
    {
        return ptr_;
    }
};

#endif // OPENCLMEMORYACCESS_H
