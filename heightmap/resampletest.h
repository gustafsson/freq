#ifndef RESAMPLETEST_H
#define RESAMPLETEST_H

#include <GpuCpuData.h>
#include <boost/scoped_ptr.hpp>

class ResampleTest
{
public:
    ResampleTest();
    ~ResampleTest();

    bool test1();
    bool test2();
private:
    template<typename T>
    void print( const char* txt, GpuCpuData<T>& data );

    void simpleData();
    void bigData( unsigned w = 1024, unsigned h = 1024 );

    boost::scoped_ptr< GpuCpuData<float2> > inputData;
};

#endif // RESAMPLETEST_H
