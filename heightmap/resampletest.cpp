#include "resampletest.h"
#include <TaskTimer.h>
#include <demangle.h>
#include <sstream>
#include <CudaException.h>

#include "resampletest.cu.h"

ResampleTest::
        ResampleTest()
{
    float2 data[] = { { 0.1f, 0.2f }, { 1.0f, 3.0f }, {0.1f, 5.5f },
                      { 5.1f, 1.2f }, { 9.0f, 2.0f }, {5.1f, 2.5f },
                      { 9.1f, -6.2f }, { 7.0f, 6.0f }, {3.1f, 7.5f }};

    inputData.reset( new GpuCpuData<float2>(
            data,
            make_uint3( 3,3,1),
            GpuCpuVoidData::CpuMemory ));
}

ResampleTest::
        ~ResampleTest()
{}


bool ResampleTest::
        test1()
{
    TaskTimer tt("ResampleTest::test1()");
    try
    {
        GpuCpuData<float2> outputData(
                0,
                make_uint3( 3,3,1) );
        memset( outputData.getCpuMemory(), 0, outputData.getSizeInBytes1D() );

        simple_resample2d(
                inputData->getCudaGlobal(),
                outputData.getCudaGlobal());

        CudaException_ThreadSynchronize();

        print( "inputData",  *inputData );
        print( "outputData", outputData );
    } catch (std::exception const& x) {
        tt.info("Caught exception: \n%s: %s", typeid(x).name(), x.what());
        return false;
    }

    return true;
}


std::ostream& operator<<(std::ostream& os, float2 v )
{
    return os << v.x << " + " << v.y << "i";
}

template<typename T>
void ResampleTest::
        print( const char* txt, GpuCpuData<T>& data )
{
    cudaExtent sz = data.getNumberOfElements();
    TaskTimer tt("%s, data size = (%lu, %lu, %lu)",
                 txt, sz.width, sz.height, sz.depth );

    for (unsigned y = 0; y < sz.height; y++ )
    {
        std::stringstream ss;

        for (unsigned x = 0; x < sz.width; x++ )
            ss << "  \t" << data.getCpuMemory()[ x + y*sz.width ];

        tt.getStream() << ss.str();
    }
}
