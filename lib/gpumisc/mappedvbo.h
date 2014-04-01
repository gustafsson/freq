#ifndef MAPPEDVBO_H
#define MAPPEDVBO_H

#include "mappedvbovoid.h"

template<typename T>
class MappedVbo: private MappedVboVoid
{
public:
    MappedVbo( pVbo vbo, DataStorageSize size )
    :   MappedVboVoid( vbo ),
        data( new DataStorage<T>(size) )
    {
        map( data.get() );
    }


    ~MappedVbo()
    {
        unmap( data.get() );
    }


    typename DataStorage<T>::ptr data;
};

#endif // MAPPEDVBO_H
