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
            out[x] = in[redundantWidth - x];
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


Tfr::pChunk Stft::
        computeWithOoura(Signal::pBuffer b)
{
    DataStorageSize actualSize(
            _window_size/2 + 1,
            b->number_of_samples()/_window_size );

    DataStorageSize n = actualSize.width * actualSize.height;

    if (0==actualSize.height) // not enough data
        return Tfr::pChunk();

    Tfr::pChunk chunk( new Tfr::StftChunk(_window_size) );

    chunk->transform_data.reset( new Tfr::ChunkData( n ));

    chunk->freqAxis = freqAxis( b->sample_rate );
    chunk->chunk_offset = b->sample_offset + _window_size/2;
    chunk->first_valid_sample = 0;
    chunk->sample_rate = b->sample_rate / _window_size;
    ((StftChunk*)chunk.get())->original_sample_rate = b->sample_rate;
    chunk->n_valid_samples = (chunk->nSamples()-1) * _window_size + 1;

    if (0 == b->sample_offset)
    {
        chunk->n_valid_samples += chunk->chunk_offset;
        chunk->chunk_offset = 0;
    }

    float* input;
    Tfr::ChunkElement* output;
    if (!b->waveform_data()->HasValidContent<CpuMemoryStorage>())
    {
        TIME_STFT TaskTimer tt("fetch input from Cpu to Gpu, %g MB", b->waveform_data()->getSizeInBytes1D()/1024.f/1024.f);
        input = CpuMemoryStorage::ReadOnly<1>( b->waveform_data() ).ptr();
        TIME_STFT ComputationSynchronize();
    }
    else
    {
        input = CpuMemoryStorage::ReadOnly<1>( b->waveform_data() ).ptr();
    }
    output = CpuMemoryStorage::WriteAll<1>( chunk->transform_data ).ptr();

    // Transform signal

    TIME_STFT TaskTimer tt2("Stft::operator compute");

    Fft ft( false );
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

    if (false)
    {
        Signal::pBuffer breal = b;
        Signal::pBuffer binv = inverse( chunk );
        float* binv_p = binv->waveform_data()->getCpuMemory();
        float* breal_p = breal->waveform_data()->getCpuMemory();
        Signal::IntervalType breal_length = breal->number_of_samples();
        Signal::IntervalType binv_length = binv->number_of_samples();
        BOOST_ASSERT( breal_length = binv_length );
        float maxd = 0;
        for(Signal::IntervalType i =0; i<breal_length; i++)
        {
            float d = breal_p[i]-binv_p[i];
            if (d*d > maxd)
                maxd = d*d;
        }

        TaskInfo("Difftest %s (value %g)", maxd<1e-8?"passed":"failed", maxd);
    }

    ComputationSynchronize();
    ComputationCheckError();

    return chunk;
}


Tfr::pChunk Stft::
        computeRedundantWithOoura(Signal::pBuffer breal)
{
    ComplexBuffer b(*breal);

    BOOST_ASSERT( 0!=_window_size );

    DataStorageSize n(
            _window_size,
            b.number_of_samples()/_window_size );

    if (0==n.height) // not enough data
        return Tfr::pChunk();

    if (32768<n.height) // TODO can't handle this
        n.height = 32768;

    Tfr::pChunk chunk( new Tfr::StftChunk() );

    chunk->transform_data.reset( new ChunkData( n ));


    Tfr::ChunkElement* input = CpuMemoryStorage::ReadOnly<1>(b.complex_waveform_data()).ptr();
    Tfr::ChunkElement* output = CpuMemoryStorage::WriteAll<1>(chunk->transform_data).ptr();

    // Transform signal

    Fft ft( true );

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

    chunk->freqAxis = freqAxis( breal->sample_rate );
    chunk->chunk_offset = b.sample_offset + _window_size/2;
    chunk->first_valid_sample = 0;
    chunk->n_valid_samples = (chunk->nSamples()-1) * _window_size + 1;
    chunk->sample_rate = b.sample_rate / chunk->nScales();
    ((StftChunk*)chunk.get())->original_sample_rate = breal->sample_rate;

    if (0 == b.sample_offset)
    {
        chunk->n_valid_samples += chunk->chunk_offset;
        chunk->chunk_offset = 0;
    }

    TIME_STFT ComputationSynchronize();

    return chunk;
}


Signal::pBuffer Stft::
        inverseWithOoura(Tfr::pChunk chunk)
{
    ComputationSynchronize();
    ComputationCheckError();
    BOOST_ASSERT( chunk->nChannels() == 1 );

    const int chunk_window_size = (int)(chunk->freqAxis.max_frequency_scalar*2 + 0.5f);
    const int actualSize = chunk_window_size/2 + 1;
    int nwindows = chunk->transform_data->getNumberOfElements().width / actualSize;

    //TIME_STFT
            TaskTimer ti("Stft::inverse, chunk_window_size = %d, b = %s", chunk_window_size, chunk->getInterval().toString().c_str());

    int
            firstSample = 0;

    if (chunk->chunk_offset != 0)
        firstSample = chunk->chunk_offset - (UnsignedF)(chunk_window_size/2);

    Signal::pBuffer b(new Signal::Buffer(firstSample, nwindows*chunk_window_size, chunk->original_sample_rate));

    BOOST_ASSERT( 0!= chunk_window_size );

    if (0==nwindows) // not enough data
        return Signal::pBuffer();

    if (32768<nwindows) // TODO can't handle this
        nwindows = 32768;

    const DataStorageSize n(
            chunk_window_size,
            nwindows );

    ComputationSynchronize();
    ComputationCheckError();

    Tfr::ChunkElement* input = CpuMemoryStorage::ReadOnly<1>( chunk->transform_data ).ptr();
    float* output = CpuMemoryStorage::WriteAll<1>( b->waveform_data() ).ptr();

    ComputationSynchronize();
    ComputationCheckError();

    // Transform signal

    ComputationSynchronize();
    ComputationCheckError();

    Fft ft(false);

    for (unsigned i=0; i < n.height; ++i)
    {
        ft.computeWithOouraC2R(
                CpuMemoryStorage::BorrowPtr<Tfr::ChunkElement>( actualSize,
                                                                input + i*actualSize),
                CpuMemoryStorage::BorrowPtr<float>( n.width,
                                                    output + i*n.width)
        );
    }

    ComputationSynchronize();
    ComputationCheckError();

    stftNormalizeInverse( b->waveform_data(), n.width );

    ComputationSynchronize();
    ComputationCheckError();

    return b;
}


Signal::pBuffer Stft::
        inverseRedundantWithOoura(Tfr::pChunk chunk)
{
    BOOST_ASSERT( chunk->nChannels() == 1 );
    int
            chunk_window_size = chunk->nScales(),
            nwindows = chunk->nSamples();

    TIME_STFT TaskTimer ti("Stft::inverse, chunk_window_size = %d, b = %s", chunk_window_size, chunk->getInterval().toString().c_str());

    int
            firstSample = 0;

    if (chunk->chunk_offset != 0)
        firstSample = chunk->chunk_offset - (UnsignedF)(chunk_window_size/2);

    ComplexBuffer b(firstSample, nwindows*chunk_window_size, chunk->original_sample_rate);

    BOOST_ASSERT( 0!= chunk_window_size );

    if (0==nwindows) // not enough data
        return Signal::pBuffer();

    if (32768<nwindows) // TODO can't handle this
        nwindows = 32768;

    DataStorageSize n(
            chunk_window_size,
            nwindows );

    Tfr::ChunkElement* input = CpuMemoryStorage::ReadOnly<1>( chunk->transform_data ).ptr();
    Tfr::ChunkElement* output = CpuMemoryStorage::WriteAll<1>( b.complex_waveform_data() ).ptr();

    // Transform signal

    Fft ft(true);

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

    Signal::pBuffer realinv = b.get_real();
    stftNormalizeInverse( realinv->waveform_data(), n.width );

    TIME_STFT ComputationSynchronize();

    return realinv;
}


} // namespace Tfr
