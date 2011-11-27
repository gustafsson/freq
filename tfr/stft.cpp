#include "stft.h"
#include "stftkernel.h"
#include "complexbuffer.h"
#include "signal/buffersource.h"
#include "cpumemorystorage.h"

#include <throwInvalidArgument.h>
#include <neat_math.h>
#include <simple_math.h>
#include <computationkernel.h>

#include <boost/date_time/posix_time/posix_time.hpp>

#ifdef _MSC_VER
#include <msc_stdc.h>
#endif

//#define TIME_STFT
#define TIME_STFT if(0)

#if defined(USE_CUDA) && !defined(USE_CUFFT)
#define USE_CUFFT
#endif

#ifdef USE_CUFFT
#include <CudaProperties.h>
#endif

using namespace boost::posix_time;
using namespace boost;

namespace Tfr {


Fft::
        Fft( bool computeRedundant )
            :
            _compute_redundant( computeRedundant )
{
}


Fft::
        ~Fft()
{
}


pChunk Fft::
        forward( Signal::pBuffer real_buffer)
{
    // cufft is faster for larger ffts, but as the GPU is the bottleneck we can
    // just as well offload it some and do it on the CPU instead

    DataStorageSize input_n = real_buffer->waveform_data()->getNumberOfElements();
    DataStorageSize output_n = input_n;

    // The in-signal is padded to a power of 2 (cufft manual suggest a "multiple
    // of 2, 3, 5 or 7" but a power of one is even better) for faster fft calculations
    output_n.width = Fft::sChunkSizeG( output_n.width - 1 );

    pChunk chunk;

    // TODO choose method based on data size and locality
    if (_compute_redundant)
    {
        ComplexBuffer b( *real_buffer );

        DataStorage<std::complex<float> >::Ptr input = b.complex_waveform_data();

        chunk.reset( new StftChunk( output_n.width, true ) );
        chunk->transform_data.reset( new ChunkData( output_n ));

#ifdef USE_CUFFT
        computeWithCufft( input, chunk->transform_data, FftDirection_Forward );
#else
        computeWithOoura( input, chunk->transform_data, FftDirection_Forward );
#endif
        chunk->freqAxis.setLinear( real_buffer->sample_rate, chunk->nScales()/2 );
    }
    else
    {
        BOOST_ASSERT(output_n.width == input_n.width);

        if (output_n.width != input_n.width)
            real_buffer = Signal::BufferSource( real_buffer ).readFixedLength( Signal::Interval( real_buffer->sample_offset, real_buffer->sample_offset + output_n.width));

        chunk.reset( new StftChunk(output_n.width, false) );
        output_n.width = ((StftChunk*)chunk.get())->nScales();
        chunk->transform_data.reset( new ChunkData( output_n ));

#ifdef USE_CUFFT
        computeWithCufftR2C( real_buffer->waveform_data(), chunk->transform_data );
#else
        computeWithOouraR2C( real_buffer->waveform_data(), chunk->transform_data );
#endif
        chunk->freqAxis.setLinear( real_buffer->sample_rate, chunk->nScales() );
    }

    chunk->chunk_offset = real_buffer->sample_offset/(float)input_n.width;
    chunk->first_valid_sample = 0;
    chunk->n_valid_samples = 1;
    chunk->sample_rate = real_buffer->sample_rate/(float)input_n.width;
    chunk->original_sample_rate = real_buffer->sample_rate;
    return chunk;
}


FreqAxis Fft::
        freqAxis( float FS )
{
    FreqAxis fa;
    fa.setLinear( FS, FS/2 );
    return fa;
}


float Fft::
        displayedTimeResolution( float FS, float /*hz*/ )
{
    return 1/FS;
}


Signal::pBuffer Fft::
        backward( pChunk chunk)
{
    Signal::pBuffer r;
    float fs = chunk->original_sample_rate;
    if (((StftChunk*)chunk.get())->redundant())
    {
        unsigned scales = chunk->nScales();

        // original_sample_rate == fs * scales
        ComplexBuffer buffer( 0, scales, fs );

#ifdef USE_CUFFT
        computeWithCufft( chunk->transform_data, buffer.complex_waveform_data(), FftDirection_Backward );
#else
        computeWithOoura( chunk->transform_data, buffer.complex_waveform_data(), FftDirection_Backward );
#endif

        r = buffer.get_real();
    }
    else
    {
        unsigned scales = ((StftChunk*)chunk.get())->window_size();

        r.reset( new Signal::Buffer(0, scales, fs ));

#ifdef USE_CUFFT
        computeWithCufftC2R(chunk->transform_data, r->waveform_data());
#else
        computeWithOouraC2R(chunk->transform_data, r->waveform_data());
#endif
    }

    if ( r->number_of_samples() != chunk->n_valid_samples )
        r = Signal::BufferSource(r).readFixedLength( Signal::Interval(0, chunk->n_valid_samples ));

    r->sample_offset = chunk->getInterval().first;

    return r;
}


/// STFT

std::vector<unsigned> Stft::_ok_chunk_sizes;

Stft::
        Stft()
:
    _window_size( 1<<11 ),
    _compute_redundant(false),
    _overlap(0.f),
    _window_type(WindowType_Rectangular)
{
    compute_redundant( _compute_redundant );
}


Tfr::pChunk Stft::
        operator() (Signal::pBuffer b)
{
    TIME_STFT TaskTimer ti("Stft::operator, _window_size = %d, b = %s, computeredundant = %s",
                           _window_size, b->getInterval().toString().c_str(), compute_redundant()?"true":"false");
    DataStorage<float>::Ptr windowedInput = prepareWindow( b->waveform_data() );

    // @see compute_redundant()
    Tfr::pChunk chunk;
    if (compute_redundant())
        chunk = ChunkWithRedundant(windowedInput);
    else
        chunk = ComputeChunk(windowedInput);

    chunk->freqAxis = freqAxis( b->sample_rate );
    chunk->chunk_offset = b->sample_offset/(float)increment() + .5f;
    chunk->first_valid_sample = 0;
    chunk->n_valid_samples = chunk->nSamples()-1;
    chunk->sample_rate = b->sample_rate / increment();
    chunk->original_sample_rate = b->sample_rate;

    if (0 == b->sample_offset)
    {
        chunk->n_valid_samples += chunk->chunk_offset;
        chunk->chunk_offset = 0;
    }

    if (false)
    {
        Signal::pBuffer breal = b;
        Signal::pBuffer binv = inverse( chunk );
        float* binv_p = binv->waveform_data()->getCpuMemory();
        float* breal_p = breal->waveform_data()->getCpuMemory();
        Signal::IntervalType breal_length = breal->number_of_samples();
        Signal::IntervalType binv_length = binv->number_of_samples();
        BOOST_ASSERT( breal_length == binv_length );
        float maxd = 0;
        for(Signal::IntervalType i =0; i<breal_length; i++)
        {
            float d = breal_p[i]-binv_p[i];
            if (d*d > maxd)
                maxd = d*d;
        }

        TaskInfo("Difftest %s (value %g)", maxd<1e-9*_window_size?"passed":"failed", maxd);
    }

    TIME_STFT TaskInfo("Stft chunk %s, %s", chunk->getInterval().toString().c_str(), chunk->getInversedInterval().toString().c_str());

    return chunk;
}


Tfr::pChunk Stft::
        ComputeChunk(DataStorage<float>::Ptr inputbuffer)
{
    BOOST_ASSERT( 0!=_window_size );

    DataStorageSize actualSize(
            _window_size/2 + 1,
            inputbuffer->size().width/_window_size );

    DataStorageSize n = actualSize.width * actualSize.height;

    if (0==actualSize.height) // not enough data
        return Tfr::pChunk();

    Tfr::pChunk chunk( new Tfr::StftChunk(_window_size, false) );
    chunk->transform_data.reset( new Tfr::ChunkData( n ));

#ifdef USE_CUFFT
    computeWithCufft(inputbuffer, chunk->transform_data, actualSize);
#else
    computeWithOoura(inputbuffer, chunk->transform_data, actualSize);
#endif

    TIME_STFT ComputationSynchronize();

    return chunk;
}


Tfr::pChunk Stft::
        ChunkWithRedundant(DataStorage<float>::Ptr inputbuffer)
{
    ComplexBuffer b(inputbuffer);

    BOOST_ASSERT( 0!=_window_size );

    DataStorageSize n(
            _window_size,
            b.number_of_samples()/_window_size );

    if (0==n.height) // not enough data
        return Tfr::pChunk();

    if (32768<n.height) // TODO can't handle this
        n.height = 32768;

    Tfr::pChunk chunk( new Tfr::StftChunk(_window_size, true) );

    chunk->transform_data.reset( new ChunkData( n.width*n.height ));

#ifdef USE_CUFFT
    computeRedundantWithCufft(b.complex_waveform_data(), chunk->transform_data, n);
#else
    computeRedundantWithOoura(b.complex_waveform_data(), chunk->transform_data, n);
#endif

    TIME_STFT ComputationSynchronize();

    return chunk;
}


Signal::pBuffer Stft::
        inverse( pChunk chunk )
{
    StftChunk* stftchunk = dynamic_cast<StftChunk*>(chunk.get());
    BOOST_ASSERT( stftchunk );
    if (stftchunk->redundant())
        return inverseWithRedundant( chunk );

    ComputationSynchronize();
    ComputationCheckError();
    BOOST_ASSERT( chunk->nChannels() == 1 );

    const int chunk_window_size = stftchunk->window_size();
    const int actualSize = stftchunk->nActualScales();
    int nwindows = chunk->transform_data->size().width / actualSize;

    TIME_STFT
            TaskTimer ti("Stft::inverse, chunk_window_size = %d, b = %s, nwindows=%d",
                         chunk_window_size, chunk->getInterval().toString().c_str(), nwindows);

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

#ifdef USE_CUFFT
    inverseWithCufft( chunk->transform_data, b->waveform_data(), n );
#else
    inverseWithOoura( chunk->transform_data, b->waveform_data(), n );
#endif

    stftNormalizeInverse( b->waveform_data(), n.width );

    ComputationSynchronize();

    return b;
}


Signal::pBuffer Stft::
        inverseWithRedundant( pChunk chunk )
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

#ifdef USE_CUFFT
    inverseRedundantWithCufft(chunk->transform_data, b.complex_waveform_data(), n);
#else
    inverseRedundantWithOoura(chunk->transform_data, b.complex_waveform_data(), n);
#endif

    TIME_STFT ComputationSynchronize();

    Signal::pBuffer realinv( new Signal::Buffer(b.sample_offset, b.number_of_samples(), b.sample_rate));
    stftNormalizeInverse( b.complex_waveform_data(), realinv->waveform_data(), n.width );

    TIME_STFT ComputationSynchronize();

    return realinv;
}


FreqAxis Stft::
        freqAxis( float FS )
{
    FreqAxis fa;

    if (compute_redundant())
        fa.setLinear( FS, _window_size-1 );
    else
        fa.setLinear( FS, _window_size/2 );

    return fa;
}


float Stft::
        displayedTimeResolution( float FS, float /*hz*/ )
{
    return _window_size / FS;
}


//static unsigned absdiff(unsigned a, unsigned b)
//{
//    return a < b ? b - a : a - b;
//}


unsigned powerprod(const unsigned*bases, const unsigned*b, unsigned N)
{
    unsigned v = 1;
    for (unsigned i=0; i<N; i++)
        for (unsigned x=0; x<b[i]; x++)
            v*=bases[i];
    return v;
}

unsigned findLargestSmaller(const unsigned* bases, unsigned* a, unsigned maxv, unsigned x, unsigned n, unsigned N)
{
    unsigned i = 0;
    while(true)
    {
        a[n] = i;

        unsigned v = powerprod(bases, a, N);
        if (v >= x)
            break;

        if (n+1<N)
            maxv = findLargestSmaller(bases, a, maxv, x, n+1, N);
        else if (v > maxv)
            maxv = v;

        ++i;
    }
    a[n] = 0;

    return maxv;
}

unsigned findSmallestGreater(const unsigned* bases, unsigned* a, unsigned minv, unsigned x, unsigned n, unsigned N)
{
    unsigned i = 0;
    while(true)
    {
        a[n] = i;

        unsigned v = powerprod(bases, a, N);
        if (n+1<N)
            minv = findSmallestGreater(bases, a, minv, x, n+1, N);
        else if (v > x && (v < minv || minv==0))
            minv = v;

        if (v > x)
            break;

        ++i;
    }
    a[n] = 0;

    return minv;
}

#ifndef USE_CUFFT
unsigned Fft::
        lChunkSizeS(unsigned x, unsigned)
{
    return lpo2s(x);
}
#else
unsigned Fft::
        lChunkSizeS(unsigned x, unsigned multiple)
{
    // It's faster but less flexible to only accept powers of 2
    //return lpo2s(x);

    multiple = std::max(1u, multiple);
    BOOST_ASSERT( spo2g(multiple-1) == lpo2s(multiple+1));

    unsigned bases[]={2, 3, 5, 7};
    unsigned a[]={0, 0, 0, 0};
    unsigned N_bases = 4; // could limit to 2 bases
    unsigned x2 = multiple*findLargestSmaller(bases, a, 0, int_div_ceil(x, multiple), 0, N_bases);
    BOOST_ASSERT( x2 < x );
    return x2;
}
#endif

#ifndef USE_CUFFT
unsigned Fft::
        sChunkSizeG(unsigned x, unsigned)
{
    return spo2g(x);
}
#else
unsigned Fft::
        sChunkSizeG(unsigned x, unsigned multiple)
{
    // It's faster but less flexible to only accept powers of 2
    //return spo2g(x);

    multiple = std::max(1u, multiple);
    BOOST_ASSERT( spo2g(multiple-1) == lpo2s(multiple+1));

    unsigned bases[]={2, 3, 5, 7};
    unsigned a[]={0, 0, 0, 0};
    unsigned N_bases = 4;
    unsigned x2 = multiple*findSmallestGreater(bases, a, 0, x/multiple, 0, N_bases);
    BOOST_ASSERT( x2 > x );
    return x2;
}
#endif

unsigned oksz(unsigned x)
{
    unsigned ls = Fft::lChunkSizeS(x+1, 4);
    unsigned sg = Fft::sChunkSizeG(x-1, 4);
    if (x-ls < sg-x)
        return ls;
    else
        return sg;
}

unsigned Stft::set_approximate_chunk_size( unsigned preferred_size )
{
    //_window_size = 1 << (unsigned)floor(log2f(preferred_size)+0.5);
    _window_size = oksz( preferred_size );

    size_t free = availableMemoryForSingleAllocation();

    unsigned multiple = 0;
    multiple++; // input
    multiple++; // output
    multiple++; // overhead during computaion

    unsigned slices = 1;
    if (slices * _window_size*multiple*sizeof(Tfr::ChunkElement) > free)
    {
        unsigned max_size = free / (slices*multiple*sizeof(Tfr::ChunkElement));
        _window_size = Fft::lChunkSizeS(max_size+1, 4);
    }

    _window_size = std::max(4u, _window_size);

    return _window_size;

//    if (_ok_chunk_sizes.empty())
//        build_performance_statistics(true);

//    std::vector<unsigned>::iterator itr =
//            std::lower_bound( _ok_chunk_sizes.begin(), _ok_chunk_sizes.end(), preferred_size );

//    unsigned N1 = *itr, N2;
//    if (itr == _ok_chunk_sizes.end())
//    {
//        N2 = spo2g( preferred_size - 1 );
//        N1 = lpo2s( preferred_size + 1 );
//    }
//    else if (itr == _ok_chunk_sizes.begin())
//        N2 = N1;
//    else
//        N2 = *--itr;

//    _chunk_size = absdiff(N1, preferred_size) < absdiff(N2, preferred_size) ? N1 : N2;
//    return _chunk_size;
}


void Stft::
        compute_redundant(bool value)
{
    _compute_redundant = value;
    if (_compute_redundant)
    {
        // free unused memory
        //_handle_ctx_c2r(0,0);
        //_handle_ctx_r2c(0,0);
    }
    else
    {
        // free unused memory
        //_handle_ctx_c2c(0,0);
    }
}


void Stft::
        setWindow(WindowType type, float overlap)
{
    _window_type = type;
    _overlap = std::max(0.f, std::min(0.98f, overlap));
}


void Stft::
        compute( Tfr::ChunkData::Ptr input, Tfr::ChunkData::Ptr output, FftDirection direction )
{
#ifdef USE_CUFFT
    computeWithCufft( input, output, direction );
#else
    computeWithOoura( input, output, direction );
#endif
}


unsigned Stft::
        increment()
{
    return std::max( 1.f, std::floor(_window_size*(1.f-_overlap) + 0.5f) );
}



template<> float Stft::computeWindowValue<Stft::WindowType_Hann>( float p )         { return (1.f / 0.5f) * (0.5f  + 0.5f*cos(M_PI*p)); }
template<> float Stft::computeWindowValue<Stft::WindowType_Hamming>( float p )      { return (1.f / 0.54f) * (0.54f  + 0.46f*cos(M_PI*p)); }
template<> float Stft::computeWindowValue<Stft::WindowType_Tukey>( float p )        { return std::fabs(p) < 0.5 ? 1.f : (1.f / 0.5f) * (0.5f  + 0.5f*cos(M_PI*(std::fabs(p)*2.f-1.f))); }
template<> float Stft::computeWindowValue<Stft::WindowType_Cosine>( float p )       { return 1.5708f * cos(M_PI*p*0.5f); }
template<> float Stft::computeWindowValue<Stft::WindowType_Lanczos>( float p )      { return 4.4305f * sin(M_PI*p)/(M_PI*p); }
template<> float Stft::computeWindowValue<Stft::WindowType_Triangular>( float p )   { return 2.f * (1.f - fabs(p)); }
template<> float Stft::computeWindowValue<Stft::WindowType_Gaussian>( float p )     { return 2.42375107349f*exp2f(-6.492127684f*p*p); } // sigma = 1/3
template<> float Stft::computeWindowValue<Stft::WindowType_BarlettHann>( float p )  { return 2.f*(0.62f-0.24f*fabs(p)+0.38f*cos(M_PI*p)); }
template<> float Stft::computeWindowValue<Stft::WindowType_Blackman>( float p )     { return 2.3809f * (0.42f + 0.5f*cos(M_PI*p) + 0.08f*cos(2.f*M_PI*p)); }
template<> float Stft::computeWindowValue<Stft::WindowType_Nuttail>( float p )      { return 2.8108f * (0.355768f + 0.487396f*cos(M_PI*p) + 0.144232f*cos(2.f*M_PI*p) + 0.012604f*cos(3.f*M_PI*p)); }
template<> float Stft::computeWindowValue<Stft::WindowType_BlackmanHarris>( float p )  { return 2.7875f * (0.35875f + 0.48829*cos(M_PI*p) + 0.14128f*cos(2.f*M_PI*p) + 0.01168f*cos(3.f*M_PI*p)); }
template<> float Stft::computeWindowValue<Stft::WindowType_BlackmanNuttail>( float p ) { return 2.7504f * (0.3635819f + 0.4891775*cos(M_PI*p) + 0.1365995f*cos(2.f*M_PI*p) + 0.0106411f*cos(3.f*M_PI*p)); }
template<> float Stft::computeWindowValue<Stft::WindowType_FlatTop>( float p ) { return 1.0f * (1.f + 1.93f*cos(M_PI*p) + 1.29f*cos(2.f*M_PI*p) + 0.388f*cos(3.f*M_PI*p) + 0.032f*cos(4.f*M_PI*p)); }
template<Stft::WindowType> float Stft::computeWindowValue( float )                  { return 1.f; }


template<Stft::WindowType Type>
void Stft::
        prepareWindowKernel( DataStorage<float>::Ptr source, DataStorage<float>::Ptr windowedData )
{
    unsigned increment = this->increment();
    unsigned windowCount = windowedData->size().width/_window_size;

    CpuMemoryReadOnly<float, 3> in = CpuMemoryStorage::ReadOnly<3>(source);
    CpuMemoryWriteOnly<float, 3> out = CpuMemoryStorage::WriteAll<3>(windowedData);

    CpuMemoryWriteOnly<float, 3>::Position pos(0,0,0);

    std::vector<float> windowfunction(_window_size);
    float* window = &windowfunction[0];
#pragma omp parallel for
    for (int x=0;x<(int)_window_size; ++x)
    {
        float p = 2.f*(x+1)/(_window_size+1) - 1.f;
        window[x] = computeWindowValue<Type>(p);
    }

    for (pos.z=0; pos.z<source->size().depth; ++pos.z)
    {
        for (pos.y=0; pos.y<source->size().height; ++pos.y)
        {
            CpuMemoryReadOnly<float, 3>::Position readPos = pos;
#pragma omp parallel for
            for (int w=0; w<(int)windowCount; ++w)
            {
                float *o = &out.ref(pos) + w*_window_size;
                float *i = &in.ref(pos) + w*increment;

                for (unsigned x=0; x<_window_size; ++x)
                    o[x] = window[x] * i[x];
            }
        }
    }
}


DataStorage<float>::Ptr Stft::
        prepareWindow( DataStorage<float>::Ptr source )
{
    if (_window_type == WindowType_Rectangular && _overlap == 0.f )
        return source;

    unsigned increment = this->increment();
    unsigned windowCount = 1 + (source->size().width-_window_size) / increment; // round down
    if (source->size().width < _window_size)
        throw std::runtime_error("Stft not enough data for window function");


    DataStorage<float>::Ptr windowedData(new DataStorage<float>(windowCount*_window_size, source->size().height, source->size().depth ));

    switch(_window_type)
    {
    case WindowType_Hann:
        prepareWindowKernel<WindowType_Hann>(source, windowedData);
        break;
    case WindowType_Hamming:
        prepareWindowKernel<WindowType_Hamming>(source, windowedData);
        break;
    case WindowType_Tukey:
        prepareWindowKernel<WindowType_Tukey>(source, windowedData);
        break;
    case WindowType_Cosine:
        prepareWindowKernel<WindowType_Cosine>(source, windowedData);
        break;
    case WindowType_Lanczos:
        prepareWindowKernel<WindowType_Lanczos>(source, windowedData);
        break;
    case WindowType_Triangular:
        prepareWindowKernel<WindowType_Triangular>(source, windowedData);
        break;
    case WindowType_Gaussian:
        prepareWindowKernel<WindowType_Gaussian>(source, windowedData);
        break;
    case WindowType_BarlettHann:
        prepareWindowKernel<WindowType_BarlettHann>(source, windowedData);
        break;
    case WindowType_Blackman:
        prepareWindowKernel<WindowType_Blackman>(source, windowedData);
        break;
    case WindowType_Nuttail:
        prepareWindowKernel<WindowType_Nuttail>(source, windowedData);
        break;
    case WindowType_BlackmanHarris:
        prepareWindowKernel<WindowType_BlackmanHarris>(source, windowedData);
        break;
    case WindowType_BlackmanNuttail:
        prepareWindowKernel<WindowType_BlackmanNuttail>(source, windowedData);
        break;
    case WindowType_FlatTop:
        prepareWindowKernel<WindowType_FlatTop>(source, windowedData);
        break;
    default:
        prepareWindowKernel<WindowType_Rectangular>(source, windowedData);
        break;
    }


    return windowedData;
}


unsigned Stft::
        build_performance_statistics(bool writeOutput, float size_of_test_signal_in_seconds)
{
    _ok_chunk_sizes.clear();
    Tfr::Stft S;
    scoped_ptr<TaskTimer> tt;
#ifdef USE_CUFFT
    if(writeOutput) tt.reset( new TaskTimer("Building STFT performance statistics for %s", CudaProperties::getCudaDeviceProp().name));
#else
    if(writeOutput) tt.reset( new TaskTimer("Building STFT performance statistics for %s", "Cpu"));
#endif
    Signal::pBuffer B = boost::shared_ptr<Signal::Buffer>( new Signal::Buffer( 0, 44100*size_of_test_signal_in_seconds, 44100 ) );
    {
        scoped_ptr<TaskTimer> tt;
        if(writeOutput) tt.reset( new TaskTimer("Filling test buffer with random data (%.1f kB or %.1f s with fs=44100)", B->number_of_samples()*sizeof(float)/1024.f, size_of_test_signal_in_seconds));

        float* p = B->waveform_data()->getCpuMemory();
        for (unsigned i = 0; i < B->number_of_samples(); i++)
            p[i] = rand() / (float)RAND_MAX;
    }

    time_duration fastest_time;
    unsigned fastest_size = 0;
    unsigned ok_size = 0;
    Tfr::pChunk C;
    time_duration latest_time[4];
    unsigned max_base = 3;
    //double base[] = {2,3,5,7};
    double base[] = {2};
    for (unsigned n = 128; n < B->number_of_samples(); n++ )
    {
        unsigned N = -1;
        unsigned selectedBase = 0;

        for (unsigned b=0; b<sizeof(base)/sizeof(base[0]) && b<=max_base; b++)
        {
            double x = ceil(log((double)n)/log(base[b]));
            unsigned N2 = pow(base[b], x);

            if (N2<N)
            {
                selectedBase = b;
                N = N2;
            }
        }

        n = N;

        S._window_size = N;

        {
            scoped_ptr<TaskTimer> tt;
            if(writeOutput) tt.reset( new TaskTimer( "n=%u, _chunk_size = %u = %g ^ %g",
                                                     n, S._window_size,
                                                     base[selectedBase],
                                                     log2f((float)S._window_size)/log2f(base[selectedBase])));

            ptime startTime = microsec_clock::local_time();

            C = S( B );

            if (!C)
                continue;

            time_duration diff = microsec_clock::local_time() - startTime;

            if (0<selectedBase)
                if (diff.total_microseconds() > latest_time[0].total_microseconds()*1.5)
                {
                    max_base = selectedBase-1;
                    if(tt) tt->info("max_base = %u", max_base);
                }

            latest_time[selectedBase] = diff;

            if (diff.total_milliseconds() < fastest_time.total_milliseconds()*1.2)
                ok_size = S._window_size;

            if (diff < fastest_time || 0==fastest_size)
            {
                max_base = sizeof(base)/sizeof(base[0]) - 1;
                fastest_time = diff;
                fastest_size = S._window_size;
            }

            _ok_chunk_sizes.push_back( S._window_size );
        }
        C.reset();

        if (S._window_size > B->number_of_samples())
            break;
    }

    if(writeOutput) TaskInfo("Fastest size = %u", fastest_size);
    return fastest_size;
}


StftChunk::
        StftChunk(unsigned window_size, bool redundant)
            :
            Chunk( Order_column_major ),
            _halfs_n( 0 ),
            _window_size( window_size ),
            _redundant( redundant )
{}


void StftChunk::
        setHalfs( unsigned n )
{
    sample_rate /= 1<<_halfs_n;

    _halfs_n = n;

    sample_rate *= 1<<_halfs_n;
}


unsigned StftChunk::
        halfs( )
{
    return _halfs_n;
}


unsigned StftChunk::
        nActualScales() const
{
    if (_redundant)
        return _window_size;
    return _window_size/2 + 1;
}


unsigned StftChunk::
        nSamples() const
{
    return transform_data->size().width / nActualScales();
}


unsigned StftChunk::
        nScales() const
{
    unsigned s = transformSize();
    if (_redundant)
        return s;
    else
        return s/2 + 1;
}


unsigned StftChunk::
        transformSize() const
{
    return _window_size >> _halfs_n;
}


} // namespace Tfr
