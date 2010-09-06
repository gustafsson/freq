#include "writewav.h"

#ifdef _MSC_VER
typedef __int64 __int64_t;
#else
#include <stdint.h> // defines __int64_t which is expected by sndfile.h
#endif

#include <sndfile.hh> // for writing various formats
#include <boost/foreach.hpp>

//#define TIME_WRITEWAV
#define TIME_WRITEWAV if(0)

namespace Signal {

WriteWav::
        WriteWav( std::string filename )
:   _data(SinkSource::AcceptStrategy_ACCEPT_EXPECTED_ONLY),
    _filename(filename)
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
    TIME_WRITEWAV TaskTimer tt("WriteWav::put [%u,%u]", buffer->sample_offset, buffer->sample_offset+buffer->number_of_samples());

    _data.put( buffer );

    if (_data.expected_samples().isEmpty())
        reset(); // Write to file
}

void WriteWav::
        reset()
{
    if (!_data.empty())
        writeToDisk();

    _data.reset();
}

bool WriteWav::
        isFinished()
{
    return expected_samples().isEmpty();
}

void WriteWav::onFinished()
{
    // WriteWav::onFinished doesn't do anything. WriteWav only calls
    // writeToDisk to disk once for each put after which
    // _data.expected_samples().isEmpty() is true.
    // Afterwards _data is reset.
}

void WriteWav::
        writeToDisk()
{
    Intervals sid = _data.samplesDesc();
    Interval i = sid.coveredInterval();

    BOOST_ASSERT(i.valid());

    TIME_WRITEWAV sid.print("data to write");
    pBuffer b = _data.readFixedLength( i.first, i.last-i.first );
    writeToDisk( _filename, b );
}

void WriteWav::
        writeToDisk(std::string filename, pBuffer b)
{
    std::stringstream ss;
    ss << b->getInterval();

    TIME_WRITEWAV TaskTimer tt("%s %s %s", __FUNCTION__, filename.c_str(), ss.str().c_str());

    if (Buffer::Only_Real != b->interleaved())
        b = b->getInterleaved( Buffer::Only_Real);

    // TODO: figure out a way for Sonic AWE to work with stereo sound and write stereo to disk


    const int format=SF_FORMAT_WAV | SF_FORMAT_PCM_16;
    //const int format=SF_FORMAT_WAV | SF_FORMAT_FLOAT;

    //int number_of_channels = 1;
    SndfileHandle outfile(filename.c_str(), SFM_WRITE, format, 1, b->sample_rate);

    if (!outfile) return;

    float *data=b->waveform_data->getCpuMemory();
    unsigned N = b->number_of_samples();

    { // Normalize

        float high=0, low=0;
        for (unsigned k=0; k<N; k++) {
            if (data[k]>high) high = data[k];
            if (data[k]<low) low = data[k];
        }

        for (unsigned k=0; k<N; k++) {
            float v = (data[k]-low)/(high-low)*2-1;
            if (v>1) v = 1;
            if (v<-1) v = -1;
            data[k] = v;
        }
    }

    outfile.write( data, N); // yes write float
}

} // namespace Signal
