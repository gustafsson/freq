#include "stft.h"
#include "stftkernel.h"
#include "complexbuffer.h"
#include "signal/buffersource.h"
#include "cpumemorystorage.h"

#include <throwInvalidArgument.h>
#include <neat_math.h>
#include <computationkernel.h>

#include <boost/date_time/posix_time/posix_time.hpp>

#ifdef _MSC_VER
#include <msc_stdc.h>
#endif

//#define TIME_STFT
#define TIME_STFT if(0)

//#define TEST_FT_INVERSE
#define TEST_FT_INVERSE if(0)

#if defined(USE_CUDA) && !defined(USE_CUFFT)
#define USE_CUFFT
#endif

#ifdef USE_CUFFT
#include "CudaProperties.h"
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
    TIME_STFT TaskTimer tt("Fft::forward %s", real_buffer->getInterval().toString().c_str());

    // cufft is faster for larger ffts, but as the GPU is the bottleneck we can
    // just as well offload it some and do it on the CPU instead

    DataStorageSize input_n = real_buffer->waveform_data()->size();
    DataStorageSize output_n = input_n;

    // The in-signal is padded to a power of 2 (cufft manual suggest a "multiple
    // of 2, 3, 5 or 7" but a power of one is even better) for faster fft calculations
    output_n.width = Fft::sChunkSizeG( output_n.width - 1 );

    pChunk chunk;

    // TODO choose method based on data size and locality
    if (_compute_redundant)
    {
        Tfr::ChunkData::Ptr input( new Tfr::ChunkData( real_buffer->waveform_data()->size()));
        ::stftToComplex( real_buffer->waveform_data(), input );

        chunk.reset( new StftChunk( output_n.width, Stft::WindowType_Rectangular, output_n.width, true ) );
        chunk->transform_data.reset( new ChunkData( output_n ));

        FftImplementation::Singleton().compute( input, chunk->transform_data, FftDirection_Forward );
        chunk->freqAxis.setLinear( real_buffer->sample_rate, chunk->nScales()/2 );
    }
    else
    {
        BOOST_ASSERT(output_n.width == input_n.width);

        if (output_n.width != input_n.width)
            real_buffer = Signal::BufferSource( real_buffer ).readFixedLength( Signal::Interval( real_buffer->sample_offset.asInteger(), (real_buffer->sample_offset + output_n.width).asInteger()));

        chunk.reset( new StftChunk(output_n.width, Stft::WindowType_Rectangular, output_n.width, false) );
        output_n.width = ((StftChunk*)chunk.get())->nScales();
        chunk->transform_data.reset( new ChunkData( output_n ));

        FftImplementation::Singleton().computeR2C( real_buffer->waveform_data(), chunk->transform_data );
        chunk->freqAxis.setLinear( real_buffer->sample_rate, chunk->nScales() );
    }

    chunk->chunk_offset = real_buffer->sample_offset/(float)input_n.width;
    chunk->first_valid_sample = 0;
    chunk->n_valid_samples = 1;
    chunk->sample_rate = real_buffer->sample_rate/(float)input_n.width;
    chunk->original_sample_rate = real_buffer->sample_rate;

    TIME_STFT ComputationSynchronize();

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


unsigned Fft::
        next_good_size( unsigned current_valid_samples_per_chunk, float /*sample_rate*/ )
{
    size_t maxsize = std::min( (size_t)(64<<20), (size_t)availableMemoryForSingleAllocation() );
    maxsize /= 3*sizeof(ChunkElement);
    return std::min(maxsize, (size_t)Fft::sChunkSizeG(current_valid_samples_per_chunk));
}


unsigned Fft::
        prev_good_size( unsigned current_valid_samples_per_chunk, float /*sample_rate*/ )
{
    size_t maxsize = std::min( (size_t)(64<<20), (size_t)availableMemoryForSingleAllocation() );
    maxsize /= 3*sizeof(ChunkElement);
    return std::min(maxsize, (size_t)Fft::lChunkSizeS(current_valid_samples_per_chunk));
}


std::string Fft::
        toString()
{
    std::stringstream ss;
    ss << "Tfr::Fft"
       << ", redundant=" << _compute_redundant;
    return ss.str();
}

Signal::pBuffer Fft::
        backward( pChunk chunk)
{
    TIME_STFT TaskTimer tt("Fft::backward %s", chunk->getInterval().toString().c_str());

    Signal::pBuffer r;
    float fs = chunk->original_sample_rate;
    if (((StftChunk*)chunk.get())->redundant())
    {
        unsigned scales = chunk->nScales();

        // original_sample_rate == fs * scales
        Tfr::ChunkData::Ptr output( new Tfr::ChunkData( scales ));

        FftImplementation::Singleton().compute( chunk->transform_data, output, FftDirection_Inverse );

        r.reset( new Signal::Buffer(0, scales, fs ));
        ::stftDiscardImag( output, r->waveform_data() );
    }
    else
    {
        unsigned scales = ((StftChunk*)chunk.get())->window_size();

        r.reset( new Signal::Buffer(0, scales, fs ));

        FftImplementation::Singleton().computeC2R( chunk->transform_data, r->waveform_data() );
    }

    unsigned original_sample_count = chunk->original_sample_rate/chunk->sample_rate + .5f;
    if ( r->number_of_samples() != original_sample_count )
        r = Signal::BufferSource(r).readFixedLength( Signal::Interval(0, original_sample_count ));

    r->sample_offset = chunk->getInterval().first;

    TIME_STFT ComputationSynchronize();

    return r;
}


unsigned Fft::
        lChunkSizeS(unsigned x, unsigned m)
{
    return FftImplementation::Singleton().lChunkSizeS(x,m);
}

unsigned Fft::
        sChunkSizeG(unsigned x, unsigned m)
{
    return FftImplementation::Singleton().sChunkSizeG(x,m);
}

/// STFT

std::vector<unsigned> Stft::_ok_chunk_sizes;

Stft::
        Stft()
:
    _window_size( 1<<11 ),
    _compute_redundant(false),
    _averaging(1),
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


    if (1 != _averaging)
    {
        unsigned width = chunk->nScales();
        unsigned height = chunk->nSamples()/_averaging;

        Tfr::ChunkData::Ptr averagedOutput(
                new Tfr::ChunkData( height*width ));

        stftAverage( chunk->transform_data, averagedOutput, width );

        chunk->transform_data = averagedOutput;
    }

    chunk->freqAxis = freqAxis( b->sample_rate );
    double increment = this->increment();
    double alignment = _window_size;
    chunk->chunk_offset = (b->sample_offset + (alignment/2 - increment/2))/(increment*_averaging);
    // chunk->first_valid_sample only makes sense if the transform is invertible, which it isn't if _averaging!=1
    chunk->first_valid_sample = ceil((alignment/2 - increment/2)/increment);
    int nSamples = chunk->nSamples();
    if (nSamples > 2*chunk->first_valid_sample)
        chunk->n_valid_samples = nSamples - 2*chunk->first_valid_sample;
    else
        chunk->n_valid_samples = 0;
    chunk->sample_rate = b->sample_rate / (increment*_averaging);
    chunk->original_sample_rate = b->sample_rate;

    TEST_FT_INVERSE
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

    TIME_STFT TaskInfo("Stft chunk %s, %s, %s",
                       chunk->getInterval().toString().c_str(),
                       chunk->getInversedInterval().toString().c_str(),
                       chunk->getCoveredInterval().toString().c_str());

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

    Tfr::pChunk chunk( new Tfr::StftChunk(_window_size, _window_type, increment(), false) );
    chunk->transform_data.reset( new Tfr::ChunkData( n ));

    FftImplementation::Singleton().compute( inputbuffer, chunk->transform_data, DataStorageSize(_window_size, actualSize.height) );

    TIME_STFT ComputationSynchronize();

    return chunk;
}


Tfr::pChunk Stft::
        ChunkWithRedundant(DataStorage<float>::Ptr inputbuffer)
{
    Tfr::ChunkData::Ptr input( new Tfr::ChunkData( inputbuffer->size()));
    ::stftToComplex( inputbuffer, input );

    BOOST_ASSERT( 0!=_window_size );

    DataStorageSize n(
            _window_size,
            inputbuffer->size().width/_window_size );

    if (0==n.height) // not enough data
        return Tfr::pChunk();

    if (32768<n.height) // TODO can't handle this
        n.height = 32768;

    Tfr::pChunk chunk( new Tfr::StftChunk(_window_size, _window_type, increment(), true) );

    chunk->transform_data.reset( new ChunkData( n.width*n.height ));

    FftImplementation::Singleton().compute( input, chunk->transform_data, n, FftDirection_Forward );

    TIME_STFT ComputationSynchronize();

    return chunk;
}


Signal::pBuffer Stft::
        inverse( pChunk chunk )
{
    BOOST_ASSERT( _averaging == 1 );

    StftChunk* stftchunk = dynamic_cast<StftChunk*>(chunk.get());
    BOOST_ASSERT( stftchunk );
    BOOST_ASSERT( 0<stftchunk->n_valid_samples );
    if (stftchunk->redundant())
        return inverseWithRedundant( chunk );

    ComputationCheckError();
    BOOST_ASSERT( chunk->nChannels() == 1 );

    const int chunk_window_size = stftchunk->window_size();
    const int actualSize = stftchunk->nActualScales();
    int nwindows = chunk->transform_data->numberOfElements() / actualSize;

    TIME_STFT ComputationSynchronize();

    TIME_STFT
            TaskTimer ti("Stft::inverse, chunk_window_size = %d, b = %s, nwindows=%d",
                         chunk_window_size, chunk->getInterval().toString().c_str(), nwindows);

    int
            firstSample = 0;

    if (chunk->chunk_offset.asFloat() != 0)
        firstSample = (chunk->chunk_offset - (UnsignedF)(chunk_window_size/2)).asInteger();

    BOOST_ASSERT( 0!= chunk_window_size );

    if (0==nwindows) // not enough data
        return Signal::pBuffer();

    if (32768<nwindows) // TODO can't handle this
        nwindows = 32768;

    const DataStorageSize n(
            chunk_window_size,
            nwindows );

    DataStorage<float>::Ptr windowedOutput(new DataStorage<float>(nwindows*chunk_window_size));

    FftImplementation::Singleton().inverse( chunk->transform_data, windowedOutput, n );

    TIME_STFT ComputationSynchronize();

    {
        TIME_STFT TaskTimer ti("normalizing %u elements", n.width);
        stftNormalizeInverse( windowedOutput, n.width );
        TIME_STFT ComputationSynchronize();
    }


    // TODO normalize while reducing
    // TODO reduce and prepare in kernel
    DataStorage<float>::Ptr signal = reduceWindow( windowedOutput, stftchunk );


    Signal::pBuffer b(new Signal::Buffer(stftchunk->getInterval().first, signal->numberOfElements(), chunk->original_sample_rate));
    *b->waveform_data() = *signal; // this will not copy any data as b->waveform_data() is empty


    return b;
}


Signal::pBuffer Stft::
        inverseWithRedundant( pChunk chunk )
{
    BOOST_ASSERT( chunk->nChannels() == 1 );
    int
            chunk_window_size = chunk->nScales(),
            nwindows = chunk->nSamples();

    TIME_STFT ComputationSynchronize();
    TIME_STFT TaskTimer ti("Stft::inverse, chunk_window_size = %d, b = %s", chunk_window_size, chunk->getInterval().toString().c_str());

    int
            firstSample = 0;

    if (chunk->chunk_offset.asFloat() != 0)
        firstSample = (chunk->chunk_offset - (UnsignedF)(chunk_window_size/2)).asInteger();

    BOOST_ASSERT( 0!= chunk_window_size );

    if (0==nwindows) // not enough data
        return Signal::pBuffer();

    if (32768<nwindows) // TODO can't handle this
        nwindows = 32768;

    DataStorageSize n(
            chunk_window_size,
            nwindows );

    Tfr::ChunkData::Ptr complexWindowedOutput( new Tfr::ChunkData(nwindows*chunk_window_size));

    FftImplementation::Singleton().compute( chunk->transform_data, complexWindowedOutput, n, FftDirection_Inverse );

    TIME_STFT ComputationSynchronize();

    DataStorage<float>::Ptr windowedOutput( new DataStorage<float>(nwindows*chunk_window_size));

    {
        TIME_STFT TaskTimer ti("normalizing %u elements", n.width);
        stftNormalizeInverse( complexWindowedOutput, windowedOutput, n.width );
        TIME_STFT ComputationSynchronize();
    }


    // TODO discard imaginary part while reducing
    StftChunk*stftchunk = dynamic_cast<StftChunk*>(chunk.get());
    DataStorage<float>::Ptr signal = reduceWindow( windowedOutput, stftchunk );


    Signal::pBuffer b(new Signal::Buffer(stftchunk->getInterval().first, signal->numberOfElements(), chunk->original_sample_rate));
    *b->waveform_data() = *signal; // will copy data


    return b;
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
    return 0.125f*_window_size / FS;
}


unsigned Stft::
        next_good_size( unsigned current_valid_samples_per_chunk, float /*sample_rate*/ )
{
    if ((int)current_valid_samples_per_chunk<_window_size)
        return _window_size;

    size_t maxsize = std::min( (size_t)(64<<20), (size_t)availableMemoryForSingleAllocation() );
    maxsize = std::max((size_t)_window_size, maxsize/(3*sizeof(ChunkElement)));
    unsigned alignment = _window_size*_averaging;
    return std::min((unsigned)maxsize, spo2g(align_up(current_valid_samples_per_chunk, alignment)/alignment)*alignment);
}


unsigned Stft::
        prev_good_size( unsigned current_valid_samples_per_chunk, float /*sample_rate*/ )
{
    if ((int)current_valid_samples_per_chunk<2*_window_size)
        return _window_size;

    size_t maxsize = std::min( (size_t)(64<<20), (size_t)availableMemoryForSingleAllocation() );
    maxsize = std::max((size_t)_window_size, maxsize/(3*sizeof(ChunkElement)));
    unsigned alignment = _window_size*_averaging;
    return std::min((unsigned)maxsize, lpo2s(align_up(current_valid_samples_per_chunk, alignment)/alignment)*alignment);
}


std::string Stft::
        toString()
{
    std::stringstream ss;
    ss << "Tfr::Stft, "
       << "window_size=" << _window_size
       << ", redundant=" << (_compute_redundant?"C2C":"R2C")
       << ", overlap=" << _overlap
       << ", window_type=" << windowTypeName();
    return ss.str();
}

//static unsigned absdiff(unsigned a, unsigned b)
//{
//    return a < b ? b - a : a - b;
//}


unsigned oksz(unsigned x)
{
    if (0 == x)
        x = 1;

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

    _window_size = std::max(4, _window_size);

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
        averaging(unsigned value)
{
    if (1 > value)
        value = 1;
    if (10 < value)
        value = 10;

    _averaging = value;
}


std::string Stft::
        windowTypeName(WindowType type)
{
    switch(type)
    {
    case WindowType_Rectangular: return "Rectangular";
    case WindowType_Hann: return "Hann";
    case WindowType_Hamming: return "Hamming";
    case WindowType_Tukey: return "Tukey";
    case WindowType_Cosine: return "Cosine";
    case WindowType_Lanczos: return "Lanczos";
    case WindowType_Triangular: return "Triangular";
    case WindowType_Gaussian: return "Gaussian";
    case WindowType_BarlettHann: return "Barlett-Hann";
    case WindowType_Blackman: return "Blackman";
    case WindowType_Nuttail: return "Nuttail";
    case WindowType_BlackmanHarris: return "Blackman-Harris";
    case WindowType_BlackmanNuttail: return "Blackman-Nuttail";
    case WindowType_FlatTop: return "Flat top";
    default: return "Unknown window type";
    }
}


bool Stft::
        applyWindowOnInverse(WindowType type)
{
    switch (type)
    {
    case WindowType_Rectangular: return false;
    case WindowType_Hann: return true;
    case WindowType_Hamming: return true;
    case WindowType_Tukey: return false;
    case WindowType_Cosine: return true;
    case WindowType_Lanczos: return true;
    case WindowType_Triangular: return false;
    case WindowType_Gaussian: return false;
    case WindowType_BarlettHann: return false;
    case WindowType_Blackman: return false;
    case WindowType_Nuttail: return false;
    case WindowType_BlackmanHarris: return false;
    case WindowType_BlackmanNuttail: return false;
    case WindowType_FlatTop: return false;
    default: return false;
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
    DataStorageSize size( _window_size, input->numberOfElements()/_window_size);
    TIME_STFT TaskTimer ti("Stft::compute %s, size = %d, %d",
                           direction == FftDirection_Forward ? "forward" : "inverse",
                           size.width, size.height);
    FftImplementation::Singleton().compute( input, output, size, direction );
}


unsigned Stft::
        increment()
{
    float wanted_increment = _window_size*(1.f-_overlap);

    // _window_size must be a multiple of increment for inverse to be correct
    int divs = std::max(1.f, std::floor(_window_size/wanted_increment));
    while (_window_size/divs*divs != _window_size && divs < _window_size)
    {
        int s = _window_size/divs;
        divs = (_window_size + s - 1)/s;
    }
    divs = std::min( _window_size, std::max( 1, divs ));

    return _window_size/divs;
}



template<> float Stft::computeWindowValue<Stft::WindowType_Hann>( float p )         { return 1.f  + cos(M_PI*p); }
template<> float Stft::computeWindowValue<Stft::WindowType_Hamming>( float p )      { return 0.54f  + 0.46f*cos(M_PI*p); }
template<> float Stft::computeWindowValue<Stft::WindowType_Tukey>( float p )        { return std::fabs(p) < 0.5 ? 2.f : 1.f + cos(M_PI*(std::fabs(p)*2.f-1.f)); }
template<> float Stft::computeWindowValue<Stft::WindowType_Cosine>( float p )       { return cos(M_PI*p*0.5f); }
template<> float Stft::computeWindowValue<Stft::WindowType_Lanczos>( float p )      { return p==0?1.f:sin(M_PI*p)/(M_PI*p); }
template<> float Stft::computeWindowValue<Stft::WindowType_Triangular>( float p )   { return 1.f - fabs(p); }
template<> float Stft::computeWindowValue<Stft::WindowType_Gaussian>( float p )     { return exp2f(-6.492127684f*p*p); } // sigma = 1/3
template<> float Stft::computeWindowValue<Stft::WindowType_BarlettHann>( float p )  { return 0.62f-0.24f*fabs(p)+0.38f*cos(M_PI*p); }
template<> float Stft::computeWindowValue<Stft::WindowType_Blackman>( float p )     { return 0.42f + 0.5f*cos(M_PI*p) + 0.08f*cos(2.f*M_PI*p); }
template<> float Stft::computeWindowValue<Stft::WindowType_Nuttail>( float p )      { return 0.355768f + 0.487396f*cos(M_PI*p) + 0.144232f*cos(2.f*M_PI*p) + 0.012604f*cos(3.f*M_PI*p); }
template<> float Stft::computeWindowValue<Stft::WindowType_BlackmanHarris>( float p )  { return 0.35875f + 0.48829*cos(M_PI*p) + 0.14128f*cos(2.f*M_PI*p) + 0.01168f*cos(3.f*M_PI*p); }
template<> float Stft::computeWindowValue<Stft::WindowType_BlackmanNuttail>( float p ) { return 0.3635819f + 0.4891775*cos(M_PI*p) + 0.1365995f*cos(2.f*M_PI*p) + 0.0106411f*cos(3.f*M_PI*p); }
template<> float Stft::computeWindowValue<Stft::WindowType_FlatTop>( float p ) { return 1.f + 1.93f*cos(M_PI*p) + 1.29f*cos(2.f*M_PI*p) + 0.388f*cos(3.f*M_PI*p) + 0.032f*cos(4.f*M_PI*p); }
template<Stft::WindowType> float Stft::computeWindowValue( float )                  { return 1.f; }


template<Stft::WindowType Type>
void Stft::
        prepareWindowKernel( DataStorage<float>::Ptr source, DataStorage<float>::Ptr windowedData )
{
    unsigned increment = this->increment();
    int windowCount = windowedData->size().width/_window_size;

    CpuMemoryReadOnly<float, 3> in = CpuMemoryStorage::ReadOnly<3>(source);
    CpuMemoryWriteOnly<float, 3> out = CpuMemoryStorage::WriteAll<3>(windowedData);

    CpuMemoryWriteOnly<float, 3>::Position pos(0,0,0);

    std::vector<float> windowfunction(_window_size);
    float* window = &windowfunction[0];
    float norm = 0;
    if (applyWindowOnInverse(Type))
    {
        for (int x=0;x<(int)_window_size; ++x)
        {
            float p = 2.f*(x+1)/(_window_size+1) - 1.f;
            float a = computeWindowValue<Type>(p);
            norm += a*a;
            window[x] = a;
        }
        norm = sqrt(_window_size / norm);
    }
    else
    {
        for (int x=0;x<(int)_window_size; ++x)
        {
            float p = 2.f*(x+1)/(_window_size+1) - 1.f;
            float a = computeWindowValue<Type>(p);
            norm += a;
            window[x] = a;
        }
        norm = _window_size / norm;
    }

    for (pos.z=0; pos.z<source->size().depth; ++pos.z)
    {
        for (pos.y=0; pos.y<source->size().height; ++pos.y)
        {
#pragma omp parallel for
            for (int w=0; w<windowCount; ++w)
            {
                float *o = &out.ref(pos) + w*_window_size;
                float *i = &in.ref(pos) + w*increment;

                for (int x=0; x<_window_size; ++x)
                    o[x] = window[x] * i[x] * norm;
            }
        }
    }
}


template<Stft::WindowType Type>
void Stft::
        reduceWindowKernel( DataStorage<float>::Ptr windowedSignal, DataStorage<float>::Ptr signal, const StftChunk* c )
{
    int increment = c->increment();
    int window_size = c->window_size();
    int windowCount = windowedSignal->size().width/window_size;
    float normalizeOverlap = increment/(float)window_size;
    float normalizeFft = 1.f; // 1.f/window_size;, todo normalize here while going through the data anyways
    float normalize = normalizeFft*normalizeOverlap;

    CpuMemoryReadOnly<float, 3> in = CpuMemoryStorage::ReadOnly<3>(windowedSignal);
    CpuMemoryWriteOnly<float, 3> out = CpuMemoryStorage::WriteAll<3>(signal);

    CpuMemoryWriteOnly<float, 3>::Position pos(0,0,0);

    std::vector<float> windowfunction(window_size);
    float* window = &windowfunction[0];

    float norm = 0;

    if (applyWindowOnInverse(Type))
    {
        for (int x=0;x<window_size; ++x)
        {
            float p = 2.f*(x+1)/(window_size+1) - 1.f;
            float a = computeWindowValue<Type>(p);
            norm += a*a;
            window[x] = normalize*a;
        }
        norm = sqrt(_window_size / norm);
    }
    else
    {
        for (int x=0;x<window_size; ++x)
            window[x] = normalize;

        norm = 1;
    }


    int out0 = _window_size/2 - increment/2 + c->first_valid_sample*increment;
    int N = signal->size().width;
    if (0 == c->chunk_offset.asFloat())
        out0 = 0;

    TaskInfo("signal->size().width = %u", signal->size().width);

    BOOST_ASSERT( c->n_valid_samples*increment + (0 == c->chunk_offset.asFloat() ? increment/2:0) == signal->size().width );

    for (pos.z=0; pos.z<windowedSignal->size().depth; ++pos.z)
    {
        for (pos.y=0; pos.y<windowedSignal->size().height; ++pos.y)
        {
            float *o = &out.ref(pos);
            for (int x=0; x<increment; ++x)
                if (x>=out0 && x<N+out0) o[x-out0] = 0;
            for (int x=0; x<signal->size().width; ++x)
                o[x] = 0;

// TODO figure out how to parallelize this... subsequent iterations of 'w' access overlapping regions of o which might work and might fail, depending on timing issues
//#pragma omp parallel for
            for (int w=0; w<windowCount; ++w)
            {
                float *o = &out.ref(pos);
                float *i = &in.ref(pos) + w*window_size;

                int x0 = w*increment;
                int x=0;
                for (; x<window_size; ++x)
                    //if (x0+x>=out0 && x0+x<N+out0) o[x0+x-out0] += i[x] * normalize;
                    if (x0+x>=out0 && x0+x<N+out0) o[x0+x-out0] += window[x] * i[x] * norm;

                //for (; x<window_size-increment; ++x)
                    //if (x0+x>=out0 && x0+x<N+out0) o[x0+x-out0] += window[x] * i[x] * norm;
                //for (; x<window_size; ++x)
                    //if (x0+x>=out0 && x0+x<N+out0) o[x0+x-out0] = window[x] * i[x] * norm;
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


DataStorage<float>::Ptr Stft::
        reduceWindow( DataStorage<float>::Ptr windowedSignal, const StftChunk* c )
{
    if (c->window_type() == WindowType_Rectangular && c->increment() == c->window_size() )
        return windowedSignal;

    unsigned increment = c->increment();
    unsigned window_size = c->window_size();
    unsigned windowCount = windowedSignal->size().width / window_size;
    BOOST_ASSERT( int(windowCount*c->window_size()) == windowedSignal->size().width );


    unsigned L = c->n_valid_samples*increment;
    if (0 == c->chunk_offset.asFloat())
    {
        L += increment/2;
    }
    DataStorage<float>::Ptr signal(new DataStorage<float>( L ));
    TaskInfo("increment = %u", increment);
    TaskInfo("c->n_valid_samples = %u", c->n_valid_samples);
    TaskInfo("c->n_valid_samples*increment = %u", c->n_valid_samples*increment);
    TaskInfo("L = %u", L);

    switch(c->window_type())
    {
    case WindowType_Hann:
        reduceWindowKernel<WindowType_Hann>(windowedSignal, signal, c);
        break;
    case WindowType_Hamming:
        reduceWindowKernel<WindowType_Hamming>(windowedSignal, signal, c);
        break;
    case WindowType_Tukey:
        reduceWindowKernel<WindowType_Tukey>(windowedSignal, signal, c);
        break;
    case WindowType_Cosine:
        reduceWindowKernel<WindowType_Cosine>(windowedSignal, signal, c);
        break;
    case WindowType_Lanczos:
        reduceWindowKernel<WindowType_Lanczos>(windowedSignal, signal, c);
        break;
    case WindowType_Triangular:
        reduceWindowKernel<WindowType_Triangular>(windowedSignal, signal, c);
        break;
    case WindowType_Gaussian:
        reduceWindowKernel<WindowType_Gaussian>(windowedSignal, signal, c);
        break;
    case WindowType_BarlettHann:
        reduceWindowKernel<WindowType_BarlettHann>(windowedSignal, signal, c);
        break;
    case WindowType_Blackman:
        reduceWindowKernel<WindowType_Blackman>(windowedSignal, signal, c);
        break;
    case WindowType_Nuttail:
        reduceWindowKernel<WindowType_Nuttail>(windowedSignal, signal, c);
        break;
    case WindowType_BlackmanHarris:
        reduceWindowKernel<WindowType_BlackmanHarris>(windowedSignal, signal, c);
        break;
    case WindowType_BlackmanNuttail:
        reduceWindowKernel<WindowType_BlackmanNuttail>(windowedSignal, signal, c);
        break;
    case WindowType_FlatTop:
        reduceWindowKernel<WindowType_FlatTop>(windowedSignal, signal, c);
        break;
    default:
        reduceWindowKernel<WindowType_Rectangular>(windowedSignal, signal, c);
        break;
    }

    return signal;
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
        for (int i = 0; i < B->number_of_samples(); i++)
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
    for (int n = 128; n < B->number_of_samples(); n++ )
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
        StftChunk(unsigned window_size, Stft::WindowType window_type, unsigned increment, bool redundant)
            :
            Chunk( Order_column_major ),
            _window_type( window_type ),
            _increment( increment ),
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


Signal::Interval StftChunk::
        getCoveredInterval() const
{
    double scale = original_sample_rate/sample_rate;
    Signal::Interval I(
            std::floor((chunk_offset + .5f).asFloat() * scale + 0.5),
            std::floor((chunk_offset + nSamples() - .5f).asFloat() * scale + 0.5)
    );

    return I;
}


} // namespace Tfr
