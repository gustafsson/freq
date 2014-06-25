#ifndef RESAMPLETEST_H
#define RESAMPLETEST_H

#include "datastorage.h"
#include <cuda_runtime.h>

class ResampleTest
{
public:
    ResampleTest();
    ~ResampleTest();

    bool test1();
    bool test2();
    bool test3();
    bool test4();
    bool test5();

private:
    template<typename T>
    void print( const char* txt, DataStorage<T>& data );

    void simpleData();
    void bigData( unsigned w = 1024, unsigned h = 1024 );

    DataStorage< float2 >::ptr inputData;
};

#endif // RESAMPLETEST_H
