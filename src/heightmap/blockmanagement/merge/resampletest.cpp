#ifdef USE_CUDA
#include "resampletest.h"
#include "resampletest.cu.h"
#include "cudaglobalstorage.h"

#include "tasktimer.h"
#include "demangle.h"
#include "CudaException.h"

#include <string.h>
#include <sstream>
#include <iomanip>

ResampleTest::
        ResampleTest()
{
}


void ResampleTest::
        simpleData()
{
    float2 data[] = { { 0.1f, 0.2f }, { 1.0f, 3.0f }, {0.1f, 5.5f },
                      { 5.1f, 1.2f }, { 9.0f, 2.0f }, {5.1f, 2.5f },
                      { 9.1f, -6.2f }, { 7.0f, 6.0f }, {3.1f, 7.5f }};

    inputData.reset( new DataStorage<float2>(3,3,1));

    memcpy(inputData->getCpuMemory(), data, sizeof(data));
}


void ResampleTest::
        bigData(unsigned w, unsigned h)
{
    float2* data = new float2[w*h];

    srand(0);
    for (unsigned y=0; y<h; ++y)
    {
        for (unsigned x=0; x<w; ++x)
        {
            /*data[ x + y*w ] = make_float2(
                    (rand()%10)*(rand()%10),
                    (rand()%10)*(rand()%10));*/
            data[ x + y*w ] = make_float2(
                    x+y,
                    0);
        }
    }

    inputData.reset( new DataStorage<float2>(w,h,1));

    memcpy(inputData->getCpuMemory(), data, w*h*sizeof(float2));

    delete[] data;
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
        bigData();

        unsigned N = 10;

        DataStorage< float > outputData(128,128,1);

        memset( outputData.getCpuMemory(), 0, outputData.numberOfBytes() );

        // warmup
        if (1<N)
        {
            simple_resample2d(
                    CudaGlobalStorage::ReadOnly<2>(inputData).getCudaPitchedPtr(),
                    CudaGlobalStorage::WriteAll<float,2>(&outputData).getCudaPitchedPtr());
        }

        CudaException_ThreadSynchronize();

        {
            TaskTimer tt("Kernel invocation, from (%u,%u = %g kB) to (%u,%u = %g kB)",
                         inputData->size().width,
                         inputData->size().height,
                         outputData.size().width,
                         outputData.size().height,
                         inputData->numberOfBytes()/1024.f,
                         outputData.numberOfBytes()/1024.f);

            for (unsigned i=0; i<N; ++i)
            {
                simple_resample2d(
                        CudaGlobalStorage::ReadOnly<2>(inputData).getCudaPitchedPtr(),
                        CudaGlobalStorage::WriteAll<float,2>(&outputData).getCudaPitchedPtr());
            }

            CudaException_ThreadSynchronize();

            float T = tt.elapsedTime();
            tt.info("Resampling %g GB/s (total reads and writes)",
                    (inputData->numberOfBytes() + outputData.numberOfBytes())/1024.f/1024.f/1024.f*N/T);
        }

        print( "inputData",  *inputData );
        print( "outputData", outputData );
    } catch (std::exception const& x) {
        tt.info("In %s, caught exception %s: %s", __FUNCTION__, typeid(x).name(), x.what());
        return false;
    }

    return true;
}

bool ResampleTest::
        test2()
{
    TaskTimer tt("ResampleTest::test2()");
    try
    {
        bigData(7, 7);

        unsigned N = 10;

        DataStorage<float> outputData( 6,6,1 );

        memset( outputData.getCpuMemory(), 0, outputData.numberOfBytes() );

        // warmup
        if (1<N)
        {
            simple_resample2d(
                    CudaGlobalStorage::ReadOnly<2>(inputData).getCudaPitchedPtr(),
                    CudaGlobalStorage::WriteAll<float,2>(&outputData).getCudaPitchedPtr());
        }

        CudaException_ThreadSynchronize();

        {
            TaskTimer tt("Kernel invocation, from (%u,%u = %g kB) to (%u,%u = %g kB)",
                         inputData->size().width,
                         inputData->size().height,
                         outputData.size().width,
                         outputData.size().height,
                         inputData->numberOfBytes()/1024.f,
                         outputData.numberOfBytes()/1024.f);

            for (unsigned i=0; i<N; ++i)
            {
                simple_resample2d(
                        CudaGlobalStorage::ReadOnly<2>(inputData).getCudaPitchedPtr(),
                        CudaGlobalStorage::WriteAll<float,2>(&outputData).getCudaPitchedPtr());
            }

            CudaException_ThreadSynchronize();

            float T = tt.elapsedTime();
            tt.info("Resampling %g GB/s (total reads and writes)",
                    (inputData->numberOfBytes() + outputData.numberOfBytes())/1024.f/1024.f/1024.f*N/T);
        }

        print( "inputData",  *inputData );
        print( "outputData", outputData );
    } catch (std::exception const& x) {
        tt.info("In %s, caught exception %s: %s", __FUNCTION__, typeid(x).name(), x.what());
        return false;
    }

    return true;
}


bool ResampleTest::
        test3()
{
    TaskTimer tt("ResampleTest::test3()");
    printf("\n");
    try
    {
        bigData(7, 7);

        DataStorage<float> outputData( 6,6,1 );
        DataStorage<float> outputData2( 6,6,1 );

        memset( outputData.getCpuMemory(), 0, outputData.numberOfBytes() );
        memset( outputData2.getCpuMemory(), 0, outputData2.numberOfBytes() );

        simple_resample2d(
                CudaGlobalStorage::ReadOnly<2>(inputData).getCudaPitchedPtr(),
                CudaGlobalStorage::WriteAll<float,2>(&outputData).getCudaPitchedPtr());

        simple_resample2d_2(
                CudaGlobalStorage::ReadOnly<2>(inputData).getCudaPitchedPtr(),
                CudaGlobalStorage::WriteAll<float,2>(&outputData2).getCudaPitchedPtr());

        print( "inputData",  *inputData );
        print( "outputData", outputData );
        print( "outputData2", outputData2 );
    } catch (std::exception const& x) {
        tt.info("In %s, caught exception %s: %s", __FUNCTION__,
                vartype(x).c_str(), x.what());
        return false;
    }

    return true;
}


bool ResampleTest::
        test4()
{
    TaskTimer tt("ResampleTest::%s", __FUNCTION__);
    try
    {
        bigData(6, 6);
        print( "inputData",  *inputData );
        simple_operate( CudaGlobalStorage::ReadOnly<2>(inputData).getCudaPitchedPtr() );
        print( "result",  *inputData );
    } catch (std::exception const& x) {
        tt.info("In %s, caught exception %s: %s", __FUNCTION__,
                vartype(x).c_str(), x.what());
        return false;
    }

    return true;
}


bool ResampleTest::
        test5()
{
    TaskTimer tt("ResampleTest::test5()");
    try
    {
        bigData();

        DataStorage<float> outputData( 128,128,1 );

        memset( outputData.getCpuMemory(), 0, outputData.numberOfBytes() );

        coordinatetest_resample2d(
                CudaGlobalStorage::ReadOnly<2>(inputData).getCudaPitchedPtr(),
                CudaGlobalStorage::WriteAll<float,2>(&outputData).getCudaPitchedPtr());

        print( "inputData",  *inputData );
        print( "outputData", outputData );
    } catch (std::exception const& x) {
        tt.info("In %s, caught exception %s: %s", __FUNCTION__, typeid(x).name(), x.what());
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
        print( const char* txt, DataStorage<T>& data )
{
    DataStorageSize sz = data.size();
    TaskTimer tt("%s, data size = (%lu, %lu, %lu)",
                 txt, sz.width, sz.height, sz.depth );

    for (int y = 0; y < sz.height; y++ )
    {
        if (y==4 && sz.height > 8)
        {
            tt.getStream() << "  \t...";
            y = sz.height-4;
        }

        std::stringstream ss;
        ss << std::setprecision(40);

        int x;
        for (x = 0; x < sz.width; x++ )
        {
            if (x==3 && sz.width > 7)
            {
                ss << "  \t...";
                x = sz.width-3;
            }
            ss << "  \t" << data.getCpuMemory()[ x + y*sz.width ];
        }

        tt.getStream() << ss.str();
    }
}
#endif
