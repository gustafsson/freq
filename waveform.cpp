#include "waveform.h"
#include <audiere.h> // for reading various formats
#include <sndfile.hh> // for writing wav
#include <math.h>
#include "Statistics.h"
#ifdef _MSC_VER
#include "windows.h"
#endif

#include <boost/scoped_array.hpp>
#include <boost/scoped_ptr.hpp>
#include <QThread>

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
    boost::scoped_array<char> data(new char[num_frames*frame_size*channel_count]);
    _source->read(num_frames, data.get());

    unsigned j=0;
    float* fdata = _waveformData->getCpuMemory();

    for (unsigned i=0; i<num_frames; i++)
    for (int c=0; c<channel_count; c++)
    {
        float f = 0;
        switch(frame_size) {
            case 0:
            case 1: f = data[i*channel_count + c]/127.; break;
            case 2: f = ((short*)data.get())[i*channel_count + c]/32767.; break;
            default:
                // assume signed LSB
                for (unsigned k=0; k<frame_size-1; k++) {
                    f+=((unsigned char*)data.get())[j++];
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

    //int number_of_channels = 1;
    SndfileHandle outfile(filename, SFM_WRITE, format, 1, _sample_rate);

    if (!outfile) return;

    outfile.write( _waveformData->getCpuMemory(), _waveformData->getNumberOfElements().width); // yes float
}

class SoundPlayer: public QThread {

public:
    SoundPlayer() {
        _device = OpenDevice();
    }

    void play( SampleBufferPtr sampleBuffer, float length )
    {
        this->quit();
        this->wait(1);

        _length = length;
        _sound = OpenSound(_device, sampleBuffer->openStream(), false);

        this->start();
    }

    virtual void run() {
        if( 0 != _sound.get()) {
            _sound->setVolume(1.0f);
            _sound->play();

            this->msleep( _length*1000 );
            while (_sound->isPlaying())
                this->msleep( _length*1000 );

            this->msleep( _length*1000 );
        }
    }
private:
    AudioDevicePtr _device;
    OutputStreamPtr _sound;
    float _length;
};

void Waveform::play() const {
    // create signed short representation
    unsigned num_frames = _waveformData->getNumberOfElements().width;
    unsigned channel_count = _waveformData->getNumberOfElements().height;
    float *fdata = _waveformData->getCpuMemory();
    unsigned firstNonzero = 0;
    unsigned lastNonzero = 0;
    for (unsigned f=0; f<num_frames; f++)
        for (unsigned c=0; c<channel_count; c++)
            if (fdata[f*channel_count + c])
                lastNonzero = max(lastNonzero, f);
            else if (firstNonzero==f)
                firstNonzero = f+1;

    if (firstNonzero > lastNonzero)
        return;

    boost::scoped_array<short> data( new short[(lastNonzero-firstNonzero+1) * channel_count] );

    for (unsigned f=firstNonzero; f<=lastNonzero; f++)
        for (unsigned c=0; c<channel_count; c++) {
            float rampup = min(1.f, (f-firstNonzero)*.01f);
            float rampdown = min(1.f, (lastNonzero-f)*.01f);
            data[(f-firstNonzero)*channel_count + c] = rampup*rampdown*fdata[ f + c*num_frames]*32767;
        }


    // play sound
    SampleBufferPtr sampleBuffer( CreateSampleBuffer(
            data.get(),
            lastNonzero-firstNonzero+1,
            channel_count,
            _sample_rate,
            SF_S16));

    static SoundPlayer sp;
    sp.play( sampleBuffer, num_frames / (float)_sample_rate );
}
