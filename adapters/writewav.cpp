#include "writewav.h"

#ifdef _MSC_VER
typedef __int64 __int64_t;
#else
#include <stdint.h> // defines __int64_t which is expected by sndfile.h
#endif

#include <sndfile.hh> // for writing various formats
 
#include <Statistics.h>

#include "cpumemorystorage.h"

#define TIME_WRITEWAV
//#define TIME_WRITEWAV if(0)

namespace Adapters {

WriteWav::
        WriteWav( std::string filename )
:   _filename(filename)
{
    TaskInfo("WriteWav %s", _filename.c_str());
}


WriteWav::
        ~WriteWav()
{
    TaskInfo("~WriteWav %s", _filename.c_str());
}


void WriteWav::
        set_channel(unsigned c)
{
    _data.set_channel( c );
}


void WriteWav::
        put( Signal::pBuffer buffer )
{
    TIME_WRITEWAV TaskTimer tt("WriteWav::put [%lu,%lu]", (long unsigned)buffer->sample_offset, (long unsigned)(buffer->sample_offset + buffer->number_of_samples()));

    //Statistics<float>(buffer->waveform_data());
    _data.putExpectedSamples( buffer );

    if (!_data.invalid_samples())
        writeToDisk();
}


void WriteWav::
        reset()
{
    if (!_data.empty())
        writeToDisk();

    _data.clear();
}


void WriteWav::
        invalidate_samples( const Signal::Intervals& s )
{
    _data.invalidate_samples( s );
    _data.setNumChannels( num_channels() );
}


Signal::Intervals WriteWav::
        invalid_samples()
{
    return _data.invalid_samples(); 
}


bool WriteWav::
        deleteMe()
{
    // don't delete, in case normalize is changed after the file has been written
    return false;
}


void WriteWav::
        normalize(bool v)
{
    if (_normalize == v)
        return;

    _normalize = v;

    if (!_data.invalid_samples())
        writeToDisk();
}


void WriteWav::
        writeToDisk()
{
    Signal::Interval i = _data.samplesDesc().coveredInterval();

    BOOST_ASSERT(i.count());

    TIME_WRITEWAV TaskTimer tt("Writing data %s", i.toString().c_str());
    Signal::pBuffer b = _data.readAllChannelsFixedLength( i );
    writeToDisk( _filename, b, _normalize );
}


void WriteWav::
        writeToDisk(std::string filename, Signal::pBuffer b, bool normalize)
{
    TIME_WRITEWAV TaskTimer tt("%s %s %s", __FUNCTION__, filename.c_str(),
                               b->getInterval().toString().c_str());

    const int format=SF_FORMAT_WAV | SF_FORMAT_PCM_16;
    //const int format=SF_FORMAT_WAV | SF_FORMAT_FLOAT;

    int C = b->waveform_data()->size().height;
    SndfileHandle outfile(filename.c_str(), SFM_WRITE, format, C, b->sample_rate);

    if (!outfile) return;

    Signal::IntervalType Nsamples_per_channel = b->number_of_samples();
    Signal::IntervalType N = Nsamples_per_channel*C;
    float* data = CpuMemoryStorage::ReadWrite<2>( b->waveform_data() ).ptr();

    if (normalize) // Normalize
    {
        long double mean = 0;
        for (unsigned k=0; k<N; k++)
            mean += data[k]/N;

        float high=0, low=0;
        long double var = 0;
        for (unsigned k=0; k<N; k++) {
            if (data[k]>high) high = data[k];
            if (data[k]<low) low = data[k];
            var += (data[k]-mean)*(data[k]-mean)/N;
        }

        if (0 == "Move DC")
        {
            for (unsigned k=0; k<N; k++) {
                float v = (data[k]-low)/(high-low)*2-1;
                if (v>1) v = 1;
                if (v<-1) v = -1;
                data[k] = v;
            }
        }
        else
        {
            high = std::max( high, -low );

            for (unsigned k=0; k<N; k++) {
                float v = data[k]/high;
                if (v>1) v = 1;
                if (v<-1) v = -1;
                data[k] = v;
            }
        }
    }

    std::vector<float> interleaved_data(N);
    for (Signal::IntervalType i=0; i<Nsamples_per_channel; ++i)
    {
        for (int c=0; c<C; ++c)
        {
            interleaved_data[i*C+c] = data[i + c*Nsamples_per_channel];
        }
    }

    outfile.write( &interleaved_data[0], N ); // sndfile will convert float to short it
}


Signal::pBuffer WriteWav::
        crop(Signal::pBuffer buffer)
{
    /// Remove zeros from the beginning and end
    //GpuCpuData<float>* waveform_data = buffer->waveform_data();
    DataStorage<float>::Ptr waveform_data = buffer->waveform_data();
    unsigned num_frames = waveform_data->size().width;
    unsigned channel_count = waveform_data->size().height;
    float *fdata = CpuMemoryStorage::ReadOnly<2>( waveform_data ).ptr();

    long unsigned firstNonzero = 0;
    long unsigned lastNonzero = 0;

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
