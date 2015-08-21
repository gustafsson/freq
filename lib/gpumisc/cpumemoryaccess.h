#ifndef CPUMEMORYACCESS_H
#define CPUMEMORYACCESS_H

#include "datastorageaccess.h"

template<typename DataType, unsigned Dimension>
class CpuMemoryAccess
{
    // implementes DataStorageAccessConcept

    DataType* ptr_;
    DataAccessSize<Dimension> sz; // numberOfElements


public:
    typedef DataType T;
    typedef DataAccessPosition<Dimension> Position;
    typedef DataAccessSize<Dimension> Size;


    CpuMemoryAccess(T* p, DataAccessSize<Dimension> numberOfElements)
        : ptr_(p),
          sz( numberOfElements )
    {
    }


    DataAccessSize<Dimension> numberOfElements() { return sz; }


    // clamps 'p' to a valid position
    T& ref( Position p )
    {
        return ptr_[ sz.offset(p) ];
    }


    T& r( Position p )
    {
        return ptr_[ sz.o(p) ];
    }


    DataType* ptr()
    {
        return ptr_;
    }
};


template<typename DataType, unsigned Dimension>
class CpuMemoryReadOnly: public CpuMemoryAccess<DataType, Dimension>
{
    // implementes DataStorageAccessConcept
public:
    typedef CpuMemoryAccess<DataType, Dimension> Access;
    typedef typename Access::Position Position;

    CpuMemoryReadOnly(const Access& r)
        : Access(r)
    {}

    DataType read( const Position& p )
    {
        return this->ref( p );
    }
};


template<typename DataType, unsigned Dimension>
class CpuMemoryWriteOnly: public CpuMemoryAccess<DataType, Dimension>
{
    // implementes DataStorageAccessConcept
public:
    typedef CpuMemoryAccess<DataType, Dimension> Access;
    typedef typename Access::Position Position;

    CpuMemoryWriteOnly(const Access& a)
        : Access(a)
    {}

    void write( const Position& p, const DataType& v )
    {
        this->ref(p) = v;
    }
};


template<typename DataType, unsigned Dimension>
class CpuMemoryReadWrite: public CpuMemoryAccess<DataType, Dimension>
{
    // implementes DataStorageAccessConcept
public:
    typedef CpuMemoryAccess<DataType, Dimension> Access;
    typedef typename Access::Position Position;

    CpuMemoryReadWrite(const Access& a)
        : Access(a)
    {}

    DataType read( const Position& p )
    {
        return ref( p );
    }

    void write( const Position& p, const DataType& v )
    {
        this->ref(p) = v;
    }
};

#endif // CPUMEMORYACCESS_H
