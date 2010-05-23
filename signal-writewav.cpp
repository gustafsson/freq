#include "signal-writewav.h"
#include <stdint.h> // defines int64_t which is expected by sndfile.h
#ifdef _MSC_VER
typedef int64_t __int64_t;
#endif
#include <sndfile.hh> // for writing various formats
#include <boost/foreach.hpp>

namespace Signal {

WriteWav::
        WriteWav( std::string filename )
            :   _filename(filename)
{
}

WriteWav::
        ~WriteWav()
{
    reset();
}

void WriteWav::
        put( pBuffer buffer )
{
    if (!_cache.empty()) if (_cache[0]->sample_rate != buffer->sample_rate) {
        throw std::logic_error(std::string(__FUNCTION__) + " sample rate is different from previous sample rate" );
    }

    _cache.push_back( buffer );

    unsigned x = expected_samples_left();
    if (x < buffer->number_of_samples() )
        x = 0;
    else
        x -= buffer->number_of_samples();
    expected_samples_left( x );

    if (0==expected_samples_left()) {
        reset();
    }
}

void WriteWav::
        reset()
{
    if (nAccumulatedSamples())
        writeToDisk();

    _cache.clear();
    expected_samples_left(0);
}

SamplesIntervalDescriptor WriteWav::
        getMissingSamples()
{
    return SamplesIntervalDescriptor(
            0,
            nAccumulatedSamples() + expected_samples_left()
            );
}

unsigned WriteWav::
        nAccumulatedSamples()
{
    // count previous samples
    unsigned nAccumulated_samples = 0;
    BOOST_FOREACH( const pBuffer& s, _cache ) {
        nAccumulated_samples += s->number_of_samples();
    }
    return nAccumulated_samples;
}

void WriteWav::
        writeToDisk()
{
    unsigned N = nAccumulatedSamples();

    if (0==N) {
        throw std::invalid_argument( std::string(__FUNCTION__) + ": refuse to write 0 samples to disk.");
    }

    pBuffer b( new Buffer);
    b->waveform_data.reset( new GpuCpuData<float>( 0, make_cudaExtent(N,1,1) ));
    b->sample_rate = _cache[0]->sample_rate;

    float* p = b->waveform_data->getCpuMemory();
    BOOST_FOREACH( const pBuffer& s, _cache ) {
        memcpy( p, s->waveform_data->getCpuMemory(), s->waveform_data->getSizeInBytes().width );
        p += s->number_of_samples();
    }

    writeToDisk( b );
}

void WriteWav::
        writeToDisk(pBuffer b)
{
    TaskTimer tt("%s %s", __FUNCTION__, _filename.c_str());

    // TODO: figure out a way for Sonic AWE to work with stereo sound and write stereo to disk


    const int format=SF_FORMAT_WAV | SF_FORMAT_PCM_16;
    //const int format=SF_FORMAT_WAV | SF_FORMAT_FLOAT;

    //int number_of_channels = 1;
    SndfileHandle outfile(_filename.c_str(), SFM_WRITE, format, 1, b->sample_rate);

    if (!outfile) return;

    outfile.write( b->waveform_data->getCpuMemory(), b->number_of_samples()); // yes write float
}

} // namespace Signal
