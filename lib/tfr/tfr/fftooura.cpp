#include "stft.h"
#include "stftkernel.h"
#include "fftooura.h"

#include "cpumemorystorage.h"
#include "complexbuffer.h"
#include "tasktimer.h"
#include "computationkernel.h"


//#define TIME_STFT
#define TIME_STFT if(0)

//#define TIME_FFT
#define TIME_FFT if(0)

// TODO Use the proper transform for different FftTransform methods

//cdft: Complex Discrete Fourier Transform
extern "C" { void cdft(int, int, float *, int *, float *); }
//rdft: Real Discrete Fourier Transform
extern "C" { void rdft(int, int, float *, int *, float *); }
//ddct: Discrete Cosine Transform
extern "C" { void ddct(int, int, float *, int *, float *); }
//ddst: Discrete Sine Transform
extern "C" { void ddst(int, int, float *, int *, float *); }
//dfct: Cosine Transform of RDFT (Real Symmetric DFT)
extern "C" { void dfct(int, float *, float *, int *, float *); }
//dfst: Sine Transform of RDFT (Real Anti-symmetric DFT)
extern "C" { void dfst(int, float *, float *, int *, float *); }

const int magicNumber = 123456;
const bool magicCheck = true;

namespace Tfr {


void FftOoura::
        compute( Tfr::ChunkData::ptr input, Tfr::ChunkData::ptr output, FftDirection direction )
{
    bool expectPrepared = false;

    TIME_FFT TaskTimer tt("Fft Ooura");

    int N = output->size().width;
    int n = input->size().width;

    EXCEPTION_ASSERT( n == N );

    if ((int)w.size() != N/2 + magicCheck && !expectPrepared)
    {
        TIME_STFT TaskInfo("Recomputing helper vectors for Ooura fft");
        w.resize(N/2 + magicCheck);
        ip.resize(2+(1<<(int)(log2f(N+0.5)-1)) + magicCheck);
        ip[0] = 0;

        if (magicCheck)
        {
            ip.back() = magicNumber;
            w.back() = magicNumber;
        }
    }

    EXCEPTION_ASSERT( (int)w.size() == N/2 + magicCheck );

    *output = *input;
    float* q = (float*)CpuMemoryStorage::ReadWrite<1>( output ).ptr();


    {
        TIME_FFT TaskTimer tt("Computing fft(N=%u, n=%u, direction=%d)", N, n, direction);
        cdft(2*N, direction, &q[0], const_cast<int*>(&ip[0]), const_cast<float*>(&w[0]));
    }


    if (magicCheck)
    {
        EXCEPTION_ASSERT( magicNumber == ip.back() );
        EXCEPTION_ASSERT( magicNumber == w.back() );
    }
}


void FftOoura::
        computeR2C( DataStorage<float>::ptr input, Tfr::ChunkData::ptr output )
{
    int denseWidth = output->size().width;
    int redundantWidth = input->size().width;

   EXCEPTION_ASSERT( denseWidth == redundantWidth/2+1 );

    // interleave input to complex data
    Tfr::ChunkData::ptr complexinput( new Tfr::ChunkData( input->size()));
    ::stftToComplex( input, complexinput );

    // make room for full output
    Tfr::ChunkData::ptr redundantOutput( new Tfr::ChunkData( redundantWidth ));

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
        computeC2R( Tfr::ChunkData::ptr input, DataStorage<float>::ptr output )
{
    int denseWidth = input->size().width;
    int redundantWidth = output->size().width;

    EXCEPTION_ASSERT( denseWidth == redundantWidth/2+1 );

    Tfr::ChunkData::ptr redundantInput( new Tfr::ChunkData( redundantWidth, input->size().height ));

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
        compute( Tfr::ChunkData::ptr inputdata, Tfr::ChunkData::ptr outputdata, DataStorageSize n, FftDirection direction )
{
    TIME_STFT TaskTimer tt("Stft Ooura");

    Tfr::ChunkElement* input = CpuMemoryStorage::ReadOnly<1>( inputdata ).ptr();
    Tfr::ChunkElement* output = CpuMemoryStorage::WriteAll<1>( outputdata ).ptr();

    EXCEPTION_ASSERT( inputdata->numberOfBytes() >= outputdata->numberOfBytes() );

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
        compute(DataStorage<float>::ptr input, Tfr::ChunkData::ptr output, DataStorageSize n )
{
    TIME_STFT TaskTimer tt("Stft Ooura R2C");

    DataStorageSize actualSize(n.width/2 + 1, n.height);

    EXCEPTION_ASSERT( (int)output->numberOfElements()/actualSize.width == n.height );
    EXCEPTION_ASSERT( (int)input->numberOfElements()/n.width == n.height );

    // interleave input to complex data
    Tfr::ChunkData::ptr complexinput( new Tfr::ChunkData( input->size()));
    ::stftToComplex( input, complexinput );

    // make room for full output
    Tfr::ChunkData::ptr redundantOutput( new Tfr::ChunkData( n.width*actualSize.height ));

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
        inverse(Tfr::ChunkData::ptr input, DataStorage<float>::ptr output, DataStorageSize n )
{
    TIME_STFT TaskTimer tt("Stft Ooura C2R");

    int denseWidth = n.width/2+1;
    int redundantWidth = n.width;
    int batchcount1 = (int)(output->numberOfElements()/redundantWidth),
             batchcount2 = (int)(input->numberOfElements()/denseWidth);

    EXCEPTION_ASSERT( batchcount1 == batchcount2 );
    EXCEPTION_ASSERT( (denseWidth-1)*2 == redundantWidth );

    Tfr::ChunkData::ptr redundantInput( new Tfr::ChunkData( n.height*redundantWidth ));

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
