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

namespace Adapters {

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
        put( Signal::pBuffer buffer )
{
    TIME_WRITEWAV TaskTimer tt("WriteWav::put [%u,%u]", buffer->sample_offset, buffer->sample_offset+buffer->number_of_samples());

    _data.putExpectedSamples( buffer, _invalid_samples );
    _invalid_samples -= buffer->getInterval();

    if (isFinished())
        reset(); // Write to file
}

void WriteWav::
        reset()
{
    if (!_data.empty())
        writeToDisk();

    _data.reset();
}


void WriteWav::
        writeToDisk()
{
    Signal::Intervals sid = _data.samplesDesc();
    Signal::Interval i = sid.coveredInterval();

    BOOST_ASSERT(i.valid());

    TIME_WRITEWAV sid.print("data to write");
    Signal::pBuffer b = _data.readFixedLength( i );
    writeToDisk( _filename, b );
}

void WriteWav::
        writeToDisk(std::string filename, Signal::pBuffer b)
{
    std::stringstream ss;
    ss << b->getInterval();

    TIME_WRITEWAV TaskTimer tt("%s %s %s", __FUNCTION__, filename.c_str(), ss.str().c_str());

    // TODO: figure out a way for Sonic AWE to work with stereo sound as this
    // method could easily write stereo sound if pBuffer had multiple channels.

    const int format=SF_FORMAT_WAV | SF_FORMAT_PCM_16;
    //const int format=SF_FORMAT_WAV | SF_FORMAT_FLOAT;

    //int number_of_channels = 1;
    SndfileHandle outfile(filename.c_str(), SFM_WRITE, format, 1, b->sample_rate);

    if (!outfile) return;

    float *data=b->waveform_data()->getCpuMemory();
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

Signal::pBuffer WriteWav::
        crop(Signal::pBuffer buffer)
{
    /// Remove zeros from the beginning and end
    GpuCpuData<float>* waveform_data = buffer->waveform_data();
    unsigned num_frames = waveform_data->getNumberOfElements().width;
    unsigned channel_count = waveform_data->getNumberOfElements().height;
    float *fdata = waveform_data->getCpuMemory();

    unsigned firstNonzero = 0;
    unsigned lastNonzero = 0;

    for (unsigned f=0; f<num_frames; f++)
        for (unsigned c=0; c<channel_count; c++)
            if (fdata[f*channel_count + c])
                lastNonzero = f;
            else if (firstNonzero==f)
                firstNonzero = f+1;

    if (firstNonzero > lastNonzero)
        return Signal::pBuffer();

    Signal::pBuffer result(new Signal::Buffer(firstNonzero, lastNonzero-firstNonzero+1, buffer->sample_rate ));
    float *data = result->waveform_data()->getCpuMemory();


    for (unsigned f=firstNonzero; f<=lastNonzero; f++)
        for (unsigned c=0; c<channel_count; c++) {
            float u = std::min(1.f, (f-firstNonzero)/(buffer->sample_rate*0.01f));
            float d = std::min(1.f, (lastNonzero-f )/(buffer->sample_rate*0.01f));
            float rampup   = 3*u*u - 2*u*u*u;
            float rampdown = 3*d*d - 2*d*d*d;

            data[f-firstNonzero + c*num_frames] = 0.5f*rampup*rampdown*fdata[ f + c*num_frames];
        }

    return result;
}

} // namespace Adapters
