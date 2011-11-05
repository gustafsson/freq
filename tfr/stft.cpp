#include "stft.h"
#include "complexbuffer.h"
#include "signal/buffersource.h"

#include <throwInvalidArgument.h>
#include <neat_math.h>
#include <simple_math.h>
#include <computationkernel.h>

#ifdef _MSC_VER
#include <msc_stdc.h>
#endif

//#define TIME_STFT
#define TIME_STFT if(0)

#if defined(USE_CUDA) && !defined(USE_CUFFT)
//#define USE_CUFFT
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

        chunk.reset( new StftChunk );
        chunk->transform_data.reset( new ChunkData( output_n ));

#ifdef USE_CUFFT
        computeWithCufft( input, chunk->transform_data, -1);
#else
        computeWithOoura( input, chunk->transform_data, -1 );
#endif
        chunk->freqAxis.setLinear( real_buffer->sample_rate, chunk->nScales()/2 );
    }
    else
    {
        BOOST_ASSERT(output_n.width == input_n.width);

        if (output_n.width != input_n.width)
            real_buffer = Signal::BufferSource( real_buffer ).readFixedLength( Signal::Interval( real_buffer->sample_offset, real_buffer->sample_offset + output_n.width));

        chunk.reset( new StftChunk(output_n.width) );
        output_n.width = ((StftChunk*)chunk.get())->nScales();
        chunk->transform_data.reset( new ChunkData( output_n ));

#ifdef USE_CUFFT
        computeWithCufftR2C( real_buffer->waveform_data(), chunk->transform_data );
#else
        computeWithOouraR2C( real_buffer->waveform_data(), chunk->transform_data );
#endif
        chunk->freqAxis.setLinear( real_buffer->sample_rate, chunk->nScales() );
    }

    chunk->chunk_offset = real_buffer->sample_offset;
    chunk->first_valid_sample = 0;
    chunk->n_valid_samples = input_n.width;
    chunk->sample_rate = real_buffer->sample_rate / ((StftChunk*)chunk.get())->transformSize();
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
    if (((StftChunk*)chunk.get())->window_size == (unsigned)-1)
    {
        unsigned scales = chunk->nScales();

        // original_sample_rate == fs * scales
        ComplexBuffer buffer( 0, scales, fs );

#ifdef USE_CUFFT
        computeWithCufft(chunk->transform_data, buffer.complex_waveform_data(), 1);
#else
        computeWithOoura(chunk->transform_data, buffer.complex_waveform_data(), 1);
#endif

        r = buffer.get_real();
    }
    else
    {
        unsigned scales = ((StftChunk*)chunk.get())->window_size;

        r.reset( new Signal::Buffer(0, scales, fs ));

#ifdef USE_CUFFT
        computeWithCufftC2R(chunk->transform_data, r->waveform_data());
#else
        computeWithOouraC2R(chunk->transform_data, r->waveform_data());
#endif
    }

    if ( r->number_of_samples() != chunk->n_valid_samples )
        r = Signal::BufferSource(r).readFixedLength( Signal::Interval(0, chunk->n_valid_samples ));

    r->sample_offset = chunk->chunk_offset;

    return r;
}


/// STFT

std::vector<unsigned> Stft::_ok_chunk_sizes;

Stft::
        Stft()
:
    _window_size( 1<<11 ),
    _compute_redundant(false)
{
    compute_redundant( _compute_redundant );
}


Tfr::pChunk Stft::
        operator() (Signal::pBuffer b)
{
    // @see compute_redundant()
    if (compute_redundant())
        return ChunkWithRedundant(b);

    TIME_STFT TaskTimer ti("Stft::operator, _window_size = %d, b = %s", _window_size, b->getInterval().toString().c_str());
    BOOST_ASSERT( 0!=_window_size );

#ifdef USE_CUFFT
    Tfr::pChunk chunk = computeWithCufft(b);
#else
    Tfr::pChunk chunk = computeWithOoura(b);
#endif

    return chunk;
}


Tfr::pChunk Stft::
        ChunkWithRedundant(Signal::pBuffer breal)
{
    TIME_STFT TaskTimer ti("Stft::ChunkWithRedundant, _window_size = %d, b = %s", _window_size, breal->getInterval().toString().c_str());

#ifdef USE_CUFFT
    Tfr::pChunk chunk = computeRedundantWithCufft(breal);
#else
    Tfr::pChunk chunk = computeRedundantWithOoura(breal);
#endif

    return chunk;
}


Signal::pBuffer Stft::
        inverse( pChunk chunk )
{

    if (compute_redundant())
        return inverseWithRedundant( chunk );

#ifdef USE_CUFFT
    Signal::pBuffer b = inverseWithCufft( chunk );
#else
    Signal::pBuffer b = inverseWithOoura( chunk );
#endif

    return b;
}


Signal::pBuffer Stft::
        inverseWithRedundant( pChunk chunk )
{
#ifdef USE_CUFFT
    Signal::pBuffer realinv = inverseRedundantWithCufft(chunk);
#else
    Signal::pBuffer realinv = inverseRedundantWithOoura(chunk);
#endif

    return realinv;
}


FreqAxis Stft::
        freqAxis( float FS )
{
    FreqAxis fa;
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


unsigned Stft::build_performance_statistics(bool writeOutput, float size_of_test_signal_in_seconds)
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
        StftChunk(unsigned window_size)
            :
            Chunk( Order_column_major ),
            window_size(window_size),
            halfs_n(0)
{}


void StftChunk::
        setHalfs( unsigned n )
{
    chunk_offset <<= halfs_n;
    n_valid_samples <<= halfs_n;

    halfs_n = n;

    chunk_offset >>= halfs_n;
    n_valid_samples >>= halfs_n;
}


unsigned StftChunk::
        halfs( )
{
    return halfs_n;
}


unsigned StftChunk::
        nActualScales() const
{
    if (window_size == (unsigned)-1)
        return Chunk::nScales();
    return window_size/2 + 1;
}


unsigned StftChunk::
        nSamples() const
{
    if (window_size == (unsigned)-1)
        return Chunk::nSamples();
    return transform_data->getNumberOfElements().width / nActualScales();
}


unsigned StftChunk::
        nScales() const
{
    unsigned s = transformSize();
    if (window_size == (unsigned)-1)
        return s;
    else
        return s/2 + 1;
}


unsigned StftChunk::
        transformSize() const
{
    if (window_size == (unsigned)-1)
        return Chunk::nScales() >> halfs_n;
    else
        return window_size >> halfs_n;
}


} // namespace Tfr
