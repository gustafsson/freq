#include "stft.h"
#include "stftkernel.h"
#include "fftooura.h"

#include "cpumemorystorage.h"
#include "complexbuffer.h"
#include "TaskTimer.h"
#include "computationkernel.h"

//#define TIME_STFT
#define TIME_STFT if(0)

// TODO translate cdft to take floats instead of doubles
//extern "C" { void cdft(int, int, double *); }
//extern "C" { void cdft(int, int, double *, int *, double *); }
extern "C" { void cdft(int, int, float *, int *, float *); }

const int magicNumber = 123456;
const bool magicCheck = true;

namespace Tfr {


void FftOoura::
        compute( Tfr::ChunkData::Ptr input, Tfr::ChunkData::Ptr output, FftDirection direction )
{
    bool expectPrepared = false;

    TIME_STFT TaskTimer tt("Fft Ooura");

    int N = output->getNumberOfElements().width;
    int n = input->getNumberOfElements().width;

    BOOST_ASSERT( n == N );

    if ((int)w.size() != N/2 + magicCheck && !expectPrepared)
    {
        TIME_STFT TaskInfo("Recopmuting helper vectors for Ooura fft");
        w.resize(N/2 + magicCheck);
        ip.resize(2+(1<<(int)(log2f(N+0.5)-1)) + magicCheck);
        ip[0] = 0;

        if (magicCheck)
        {
            ip.back() = magicNumber;
            w.back() = magicNumber;
        }
    }

    BOOST_ASSERT( (int)w.size() == N/2 + magicCheck );

    *output = *input;
    float* q = (float*)CpuMemoryStorage::ReadWrite<1>( output ).ptr();


    {
        TIME_STFT TaskTimer tt("Computing fft(N=%u, n=%u, direction=%d)", N, n, direction);
        cdft(2*N, direction, &q[0], const_cast<int*>(&ip[0]), const_cast<float*>(&w[0]));
    }


    if (magicCheck)
    {
        BOOST_ASSERT( magicNumber == ip.back() );
        BOOST_ASSERT( magicNumber == w.back() );
    }
}


void FftOoura::
        computeR2C( DataStorage<float>::Ptr input, Tfr::ChunkData::Ptr output )
{
    int denseWidth = output->size().width;
    int redundantWidth = input->size().width;

   BOOST_ASSERT( denseWidth == redundantWidth/2+1 );

    // interleave input to complex data
    Tfr::ChunkData::Ptr complexinput( new Tfr::ChunkData( input->size()));
    ::stftToComplex( input, complexinput );

    // make room for full output
    Tfr::ChunkData::Ptr redundantOutput( new Tfr::ChunkData( redundantWidth ));

    // compute
    compute(complexinput, redundantOutput, FftDirection_Forward);

    // discard redundant output
    {
        Tfr::ChunkElement* in = CpuMemoryStorage::ReadOnly<1>( redundantOutput ).ptr();
        Tfr::ChunkElement* out = CpuMemoryStorage::WriteAll<1>( output ).ptr();
        int x;
        for (x=0; x<denseWidth; ++x)
            out[x] = in[x];
    }
}


void FftOoura::
        computeC2R( Tfr::ChunkData::Ptr input, DataStorage<float>::Ptr output )
{
    int denseWidth = input->size().width;
    int redundantWidth = output->size().width;

    BOOST_ASSERT( denseWidth == redundantWidth/2+1 );

    Tfr::ChunkData::Ptr redundantInput( new Tfr::ChunkData( redundantWidth, input->size().height ));

    {
        Tfr::ChunkElement* in = CpuMemoryStorage::ReadOnly<1>( input ).ptr();
        Tfr::ChunkElement* out = CpuMemoryStorage::WriteAll<1>( redundantInput ).ptr();
        int x;
        for (x=0; x<denseWidth; ++x)
            out[x] = in[x];
        for (; x<redundantWidth; ++x)
            out[x] = conj(in[redundantWidth - x]);
    }

    ComplexBuffer buffer( 0, redundantWidth, 1 );

    compute(redundantInput, buffer.complex_waveform_data(), FftDirection_Inverse);

    *output = *buffer.get_real()->waveform_data();
}


void FftOoura::
        compute( Tfr::ChunkData::Ptr inputdata, Tfr::ChunkData::Ptr outputdata, DataStorageSize n, FftDirection direction )
{
    Tfr::ChunkElement* input = CpuMemoryStorage::ReadOnly<1>( inputdata ).ptr();
    Tfr::ChunkElement* output = CpuMemoryStorage::WriteAll<1>( outputdata ).ptr();

    BOOST_ASSERT( inputdata->numberOfBytes() == outputdata->numberOfBytes() );

    // Transform signal

    int i=0;
    compute(
            CpuMemoryStorage::BorrowPtr<Tfr::ChunkElement>( n.width,
                                                            input + i*n.width),
            CpuMemoryStorage::BorrowPtr<Tfr::ChunkElement>( n.width,
                                                            output + i*n.width),
            direction
    );

#pragma omp parallel for
    for (i=1; i < n.height; ++i)
    {
        compute(
                CpuMemoryStorage::BorrowPtr<Tfr::ChunkElement>( n.width,
                                                                input + i*n.width),
                CpuMemoryStorage::BorrowPtr<Tfr::ChunkElement>( n.width,
                                                                output + i*n.width),
                direction
        );
    }

    TIME_STFT ComputationSynchronize();
}


void FftOoura::
        compute(DataStorage<float>::Ptr input, Tfr::ChunkData::Ptr output, DataStorageSize n )
{
    DataStorageSize actualSize(n.width/2 + 1, n.height);

    BOOST_ASSERT( (int)output->numberOfElements()/actualSize.width == n.height );
    BOOST_ASSERT( (int)input->numberOfElements()/n.width == n.height );

    // interleave input to complex data
    Tfr::ChunkData::Ptr complexinput( new Tfr::ChunkData( input->size()));
    ::stftToComplex( input, complexinput );

    // make room for full output
    Tfr::ChunkData::Ptr redundantOutput( new Tfr::ChunkData( n.width*actualSize.height ));

    // compute
    compute(complexinput, redundantOutput, n, FftDirection_Forward );

    // discard redundant output
    Tfr::ChunkElement* in = CpuMemoryStorage::ReadOnly<1>( redundantOutput ).ptr();
    Tfr::ChunkElement* out = CpuMemoryStorage::WriteAll<1>( output ).ptr();

#pragma omp parallel for
    for (int i=0; i < actualSize.height; ++i)
    {
        int x;
        for (x=0; x<actualSize.width; ++x)
            out[i*actualSize.width + x] = in[i*n.width+x];
    }
}


void FftOoura::
        inverse(Tfr::ChunkData::Ptr input, DataStorage<float>::Ptr output, DataStorageSize n )
{
    int denseWidth = n.width/2+1;
    int redundantWidth = n.width;
    int batchcount1 = output->numberOfElements()/redundantWidth,
             batchcount2 = input->numberOfElements()/denseWidth;

    BOOST_ASSERT( batchcount1 == batchcount2 );
    BOOST_ASSERT( (denseWidth-1)*2 == redundantWidth );

    Tfr::ChunkData::Ptr redundantInput( new Tfr::ChunkData( n.height*redundantWidth ));

    {
        Tfr::ChunkElement* in = CpuMemoryStorage::ReadOnly<1>( input ).ptr();
        Tfr::ChunkElement* out = CpuMemoryStorage::WriteAll<1>( redundantInput ).ptr();
#pragma omp parallel for
        for (int i=0; i<n.height; ++i)
        {
            int x;
            for (x=0; x<denseWidth; ++x)
                out[i*redundantWidth + x] = in[i*denseWidth + x];
            for (; x<redundantWidth; ++x)
                out[i*redundantWidth + x] = conj(in[i*denseWidth + redundantWidth - x]);
        }
    }

    ComplexBuffer buffer( 0, redundantWidth*n.height, 1 );

    compute(redundantInput, buffer.complex_waveform_data(), n, FftDirection_Inverse );

    *output = *buffer.get_real()->waveform_data();

    TIME_STFT ComputationSynchronize();
}


} // namespace Tfr
