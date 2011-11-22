#ifndef USE_CUDA
#include "stft.h"

#include "cpumemorystorage.h"
#include "complexbuffer.h"
#include "TaskTimer.h"
#include "computationkernel.h"

#include "waveletkernel.h"

//#define TIME_STFT
#define TIME_STFT if(0)

// TODO translate cdft to take floats instead of doubles
//extern "C" { void cdft(int, int, double *); }
extern "C" { void cdft(int, int, double *, int *, double *); }
// extern "C" { void cdft(int, int, float *, int *, float *); }


namespace Tfr {


void Fft::
        computeWithOoura( Tfr::ChunkData::Ptr input, Tfr::ChunkData::Ptr output, FftDirection direction )
{
    TIME_STFT TaskTimer tt("Fft Ooura");

    unsigned n = input->getNumberOfElements().width;
    unsigned N = output->getNumberOfElements().width;

    if (-1 != direction)
        BOOST_ASSERT( n == N );

    int magic = 12345678;
    bool vector_length_test = true;

    std::vector<double> w(N/2 + vector_length_test);
    std::vector<int> ip(2+(1<<(int)(log2f(N+0.5)-1)) + vector_length_test);
    std::vector<double> q(2*N + vector_length_test);

    ip[0] = 0;

    if (vector_length_test)
    {
        q.back() = magic;
        w.back() = magic;
        ip.back() = magic;
    }


    {
        TIME_STFT TaskTimer tt("Converting from float2 to double2" );

        float* p = (float*)input->getCpuMemory();
        for (unsigned i=0; i<2*n; i++)
            q[i] = p[i];

        for (unsigned i=2*n; i<2*N; i++)
            q[i] = 0;
    }


    {
        TIME_STFT TaskTimer tt("Computing fft(N=%u, n=%u, direction=%d)", N, n, direction);
        cdft(2*N, direction, &q[0], &ip[0], &w[0]);

        if (vector_length_test)
        {
            BOOST_ASSERT(q.back() == magic);
            BOOST_ASSERT(ip.back() == magic);
            BOOST_ASSERT(w.back() == magic);
        }
    }

    {
        TIME_STFT TaskTimer tt("Converting from double2 to float2");

        float* p = (float*)output->getCpuMemory();
        for (unsigned i=0; i<2*N; i++)
            p[i] = (float)q[i];
    }
}


void Fft::
        computeWithOouraR2C( DataStorage<float>::Ptr input, Tfr::ChunkData::Ptr output )
{
    unsigned denseWidth = output->size().width;
    unsigned redundantWidth = input->size().width;

   BOOST_ASSERT( denseWidth == redundantWidth/2+1 );

    // interleave input to complex data
    Signal::Buffer buffer(0, redundantWidth, 1 );
    //*buffer.waveform_data() = *input;
    new CpuMemoryStorage(buffer.waveform_data().get(), input->getCpuMemory());
    ComplexBuffer complexbuffer( buffer );

    // make room for full output
    Tfr::ChunkData::Ptr redundantOutput( new Tfr::ChunkData( redundantWidth ));

    // compute
    computeWithOoura(complexbuffer.complex_waveform_data(), redundantOutput, FftDirection_Forward);

    // discard redundant output
    {
        Tfr::ChunkElement* in = CpuMemoryStorage::ReadOnly<1>( redundantOutput ).ptr();
        Tfr::ChunkElement* out = CpuMemoryStorage::WriteAll<1>( output ).ptr();
        unsigned x;
        for (x=0; x<denseWidth; ++x)
            out[x] = in[x];
    }
}


void Fft::
        computeWithOouraC2R( Tfr::ChunkData::Ptr input, DataStorage<float>::Ptr output )
{
    unsigned denseWidth = input->size().width;
    unsigned redundantWidth = output->size().width;

    BOOST_ASSERT( denseWidth == redundantWidth/2+1 );

    Tfr::ChunkData::Ptr redundantInput( new Tfr::ChunkData( redundantWidth, input->size().height ));

    {
        Tfr::ChunkElement* in = CpuMemoryStorage::ReadOnly<1>( input ).ptr();
        Tfr::ChunkElement* out = CpuMemoryStorage::WriteAll<1>( redundantInput ).ptr();
        unsigned x;
        for (x=0; x<denseWidth; ++x)
            out[x] = in[x];
        for (; x<redundantWidth; ++x)
            out[x] = conj(in[redundantWidth - x]);
    }

    ComplexBuffer buffer( 0, redundantWidth, 1 );

    computeWithOoura(redundantInput, buffer.complex_waveform_data(), FftDirection_Backward);

    *output = *buffer.get_real()->waveform_data();
}


void Stft::
        computeWithOoura( Tfr::ChunkData::Ptr input, Tfr::ChunkData::Ptr output, FftDirection direction )
{
    TaskTimer tt("Stft::computeWithOoura( matrix[%d, %d], %s )",
                 input->size().width,
                 input->size().height,
                 direction==FftDirection_Forward?"forward":"backward");

    Tfr::ChunkElement* i = CpuMemoryStorage::ReadOnly<1>( input ).ptr();
    Tfr::ChunkElement* o = CpuMemoryStorage::WriteAll<1>( output ).ptr();

    BOOST_ASSERT( output->numberOfBytes() == input->numberOfBytes() );

    const int count = input->numberOfElements()/_window_size;

    Fft ft( true );

#pragma omp parallel for
    for (int n=0; n<count; ++n)
    {
        ft.computeWithOoura(
                CpuMemoryStorage::BorrowPtr<Tfr::ChunkElement>( _window_size,
                                                                i + n*_window_size),
                CpuMemoryStorage::BorrowPtr<Tfr::ChunkElement>( _window_size,
                                                                o + n*_window_size),
                direction
        );
    }
}


void Stft::
        computeWithOoura(DataStorage<float>::Ptr inputbuffer, Tfr::ChunkData::Ptr transform_data, DataStorageSize actualSize)
{
    float* input = CpuMemoryStorage::ReadOnly<1>( inputbuffer ).ptr();
    Tfr::ChunkElement* output = CpuMemoryStorage::WriteAll<1>( transform_data ).ptr();

    // Transform signal
    TIME_STFT TaskTimer tt2("Stft::operator compute");

    Fft ft( false );

#pragma omp parallel for
    for (unsigned i=0; i < actualSize.height; ++i)
    {
        ft.computeWithOouraR2C(
                CpuMemoryStorage::BorrowPtr<float>( _window_size,
                                                    input + i*_window_size),
                CpuMemoryStorage::BorrowPtr<Tfr::ChunkElement>( actualSize.width,
                                                                output + i*actualSize.width)
        );
    }

    TIME_STFT ComputationSynchronize();

    ComputationSynchronize();
    ComputationCheckError();
}


void Stft::
        computeRedundantWithOoura(Tfr::ChunkData::Ptr inputdata, Tfr::ChunkData::Ptr outputdata, DataStorageSize n)
{
    Tfr::ChunkElement* input = CpuMemoryStorage::ReadOnly<1>(inputdata).ptr();
    Tfr::ChunkElement* output = CpuMemoryStorage::WriteAll<1>(outputdata).ptr();

    // Transform signal

    Fft ft( true );

#pragma omp parallel for
    for (unsigned i=0; i < n.height; ++i)
    {
        ft.computeWithOoura(
                CpuMemoryStorage::BorrowPtr<Tfr::ChunkElement>( n.width,
                                                                input + i*n.width),
                CpuMemoryStorage::BorrowPtr<Tfr::ChunkElement>( n.width,
                                                                output + i*n.width),
                FftDirection_Forward
        );
    }
}


void Stft::
        inverseWithOoura(Tfr::ChunkData::Ptr inputdata, DataStorage<float>::Ptr outputdata, DataStorageSize n)
{
    const int actualSize = n.width/2 + 1;
    Tfr::ChunkElement* input = CpuMemoryStorage::ReadOnly<1>( inputdata ).ptr();
    float* output = CpuMemoryStorage::WriteAll<1>( outputdata ).ptr();

    // Transform signal

    Fft ft(false);

#pragma omp parallel for
    for (unsigned i=0; i < n.height; ++i)
    {
        ft.computeWithOouraC2R(
                CpuMemoryStorage::BorrowPtr<Tfr::ChunkElement>( actualSize,
                                                                input + i*actualSize),
                CpuMemoryStorage::BorrowPtr<float>( n.width,
                                                    output + i*n.width)
        );
    }

    TIME_STFT ComputationSynchronize();
}


void Stft::
        inverseRedundantWithOoura( Tfr::ChunkData::Ptr inputdata, Tfr::ChunkData::Ptr outputdata, DataStorageSize n )
{
    Tfr::ChunkElement* input = CpuMemoryStorage::ReadOnly<1>( inputdata ).ptr();
    Tfr::ChunkElement* output = CpuMemoryStorage::WriteAll<1>( outputdata ).ptr();

    // Transform signal

    Fft ft(true);

#pragma omp parallel for
    for (unsigned i=0; i < n.height; ++i)
    {
        ft.computeWithOoura(
                CpuMemoryStorage::BorrowPtr<Tfr::ChunkElement>( n.width,
                                                                input + i*n.width),
                CpuMemoryStorage::BorrowPtr<Tfr::ChunkElement>( n.width,
                                                                output + i*n.width),
                FftDirection_Backward
        );
    }

    TIME_STFT ComputationSynchronize();
}


} // namespace Tfr
#endif // #ifndef USE_CUDA
