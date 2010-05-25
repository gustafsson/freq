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
    TaskTimer tt("WriteWav::put [%u,%u]", buffer->sample_offset, buffer->sample_offset+buffer->number_of_samples());

    SinkSource::put( buffer );

    if (_expected_samples.isEmpty())
        reset(); // Write to file
}

void WriteWav::
        reset()
{
    if (!SinkSource::empty())
        writeToDisk();

    SinkSource::reset();
}

bool WriteWav::
        finished()
{
    return expected_samples().isEmpty();
}

void WriteWav::
        writeToDisk()
{
    if (SinkSource::empty()) {
        throw std::invalid_argument( std::string(__FUNCTION__) + ": refuse to write 0 samples to disk.");
    }

    SamplesIntervalDescriptor sid = samplesDesc();
    SamplesIntervalDescriptor::Interval i = sid.getInterval( SamplesIntervalDescriptor::SampleType_MAX, 0 );

    pBuffer b = SinkSource::readFixedLength( i.first, i.last );
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
