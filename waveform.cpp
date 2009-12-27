#include "waveform.h"
#include <audiere.h> // for reading various formats
#include <sndfile.hh> // for writing wav
#include <math.h>
#include "Statistics.h"

using namespace audiere;

#include <iostream>
using namespace std;

Waveform::Waveform()
:   _sample_rate(0),
    _source(0)
{}

Waveform::Waveform (const char* filename)
{
    _source = OpenSampleSource (filename); // , FileFormat file_format=FF_AUTODETECT
    if (0==_source)
        throw std::ios_base::failure(string() + "File " + filename + " not found");

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
    for (int c=0; c<channel_count; c++)
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

void Waveform::writeFile( const char* filename ) const
{
    // todo: this method only writes mono data from the first (left) channel

    const int format=SF_FORMAT_WAV | SF_FORMAT_PCM_16;
    //  const int format=SF_FORMAT_WAV | SF_FORMAT_FLOAT;

    int number_of_channels = 1;
    SndfileHandle outfile(filename, SFM_WRITE, format, 1, _sample_rate);

    if (not outfile) return;

    outfile.write( _waveformData->getCpuMemory(), _waveformData->getNumberOfElements().width); // yes float
}
