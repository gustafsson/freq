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
    TaskTimer tt("WriteWav::put [%u,%u]", buffer->sample_offset, buffer->sample_offset+buffer->number_of_samples());

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
    SamplesIntervalDescriptor sid = _data.samplesDesc();
    SamplesIntervalDescriptor::Interval i = sid.coveredInterval();

    BOOST_ASSERT(i.valid());

    sid.print("data to write");
    pBuffer b = _data.readFixedLength( i.first, i.last );
    writeToDisk( _filename, b );
}

void WriteWav::
        writeToDisk(std::string filename, pBuffer b)
{
    std::stringstream ss;
    ss << b->getInterval();

    TaskTimer tt("%s %s %s", __FUNCTION__, filename.c_str(), ss.str().c_str());

    // TODO: figure out a way for Sonic AWE to work with stereo sound and write stereo to disk


    const int format=SF_FORMAT_WAV | SF_FORMAT_PCM_16;
    //const int format=SF_FORMAT_WAV | SF_FORMAT_FLOAT;

    //int number_of_channels = 1;
    SndfileHandle outfile(filename.c_str(), SFM_WRITE, format, 1, b->sample_rate);

    if (!outfile) return;

    outfile.write( b->waveform_data->getCpuMemory(), b->number_of_samples()); // yes write float
}

} // namespace Signal
