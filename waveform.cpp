#include "waveform.h"
#include <audiere.h>
#include <math.h>
#include "Statistics.h"

using namespace audiere;

#include <iostream>
using namespace std;

Waveform::Waveform (const char* filename)
{
    _source = OpenSampleSource (filename); // , FileFormat file_format=FF_AUTODETECT
    SampleFormat sample_format;
    int channel_count;
    _source->getFormat( channel_count, _sample_rate, sample_format);


    unsigned frame_size = GetSampleSize(sample_format);
    unsigned num_frames = _source->getLength();

    _waveformData.reset( new GpuCpuData<float>(0, make_uint3( num_frames, channel_count, 1)) );
    std::vector<char> data(num_frames*frame_size*channel_count);
    _source->read(num_frames, data.data());

    unsigned j=0;
    float* fdata = _waveformData->getCpuMemory();

    for (unsigned i=0; i<num_frames; i++)
    for (unsigned c=0; c<channel_count; c++)
    {
        float f = 0;
        switch(frame_size) {
            case 0:
            case 1: f = data[i*channel_count + c]/127.; break;
            case 2: f = ((short*)data.data())[i*channel_count + c]/32767.; break;
            default:
                // assume signed LSB
                for (unsigned k=0; k<frame_size-1; k++) {
                    f+=((unsigned char*)data.data())[j++];
                    f/=256.;
                }
                f+=data[j++];
                f/=128.;
                break;
        }

        fdata[ i + c*num_frames] = f;
    }

    Statistics<float> waveform( _waveformData.get() );
}
