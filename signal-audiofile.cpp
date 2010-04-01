#include "signal-audiofile.h"
#include <audiere.h> // for reading various formats
#include <sndfile.hh> // for writing wav
#include <math.h>
#include "Statistics.h"
#include <stdexcept>
#include <iostream>
#include <sstream>

#ifdef _MSC_VER
#include "windows.h"
#endif

#include <boost/scoped_array.hpp>
#include <boost/scoped_ptr.hpp>
#include <QThread>
#include <QSound>

#if LEKA_FFT
#include <cufft.h>

static void cufftSafeCall( cufftResult_t cufftResult) {
    if (cufftResult != CUFFT_SUCCESS) {
        ThrowInvalidArgument( cufftResult );
    }
}
#endif

using namespace std;
using namespace audiere;

namespace Signal {

Audiofile::Audiofile()
:   _source(0)
{
    _waveform.reset( new Buffer());
}


/**
  Reads an audio file using libaudiere
  */
Audiofile::Audiofile(const char* filename)
{
    _waveform.reset( new Buffer());

    _source = OpenSampleSource (filename); // , FileFormat file_format=FF_AUTODETECT
    if (0==_source) {
        stringstream ss;

        std::vector<FileFormatDesc> formats;
        GetSupportedFileFormats(formats);
        ss << "Couldn't open " << filename << endl
           << endl
           << "Supported audio file formats through Audiere:";

        for (unsigned n=0; n<formats.size(); n++) {
            ss << "  " << formats[n].description << " {";
            for (unsigned k=0; k<formats[n].extensions.size(); k++) {
                if (k) ss << ", ";
                ss << formats[n].extensions[k];
            }
            ss << "}" << endl;
        }

        throw std::ios_base::failure(ss.str());
    }

    SampleFormat sample_format;
    int channel_count, sample_rate;
    _source->getFormat( channel_count, sample_rate, sample_format);
    _waveform->sample_rate=sample_rate;

    unsigned frame_size = GetSampleSize(sample_format);
    unsigned num_frames = _source->getLength();

    if (0==num_frames)
        throw std::ios_base::failure(string() + "Opened source file but failed reading data from " + filename + "\n"
                                     "\n"
                                     "Supported audio file formats through Audiere: Ogg Vorbis, MP3, FLAC, Speex, uncompressed WAV, AIFF, MOD, S3M, XM, IT");

    _waveform->waveform_data.reset( new GpuCpuData<float>(0, make_uint3( num_frames, channel_count, 1)) );
    boost::scoped_array<char> data(new char[num_frames*frame_size*channel_count]);
    _source->read(num_frames, data.get());

    unsigned j=0;
    float* fdata = _waveform->waveform_data->getCpuMemory();

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

    //_waveform = getChunk( 0, number_of_samples(), 0, Waveform_chunk::Only_Real );
    //Statistics<float> waveform( _waveform->waveform_data.get() );

#if LEKA_FFT
/* do stupid things: */
    num_frames = 1<<13;
    GpuCpuData<float> ut2(0, make_cudaExtent(3*num_frames*sizeof(float),1,1));
    GpuCpuData<float> ut(0, make_cudaExtent(3*num_frames*sizeof(float),1,1));
//    for (unsigned i=0; i<num_frames/2; i++)
//        printf("\t%g\n",fdata[i]);

    cufftHandle                             _fft_dummy;
    cufftSafeCall(cufftPlan1d(&_fft_dummy, num_frames, CUFFT_R2C, 1));
    cufftSafeCall(cufftExecR2C(_fft_dummy,
                               (cufftReal *)_waveform.waveform_data->getCudaGlobal().ptr(),
                               (cufftComplex *)ut2.getCudaGlobal().ptr()));
    cufftSafeCall(cufftPlan1d(&_fft_dummy, num_frames, CUFFT_C2C, 1));
    /*cufftSafeCall(cufftExecC2C(_fft_dummy,
                               (cufftComplex *)_waveform.waveform_data->getCudaGlobal().ptr(),
                               (cufftComplex *)ut2.getCudaGlobal().ptr(),
                               CUFFT_FORWARD));*/
    cufftSafeCall(cufftExecC2C(_fft_dummy,
                               (cufftComplex *)ut2.getCudaGlobal().ptr(),
                               (cufftComplex *)ut.getCudaGlobal().ptr(),
                               CUFFT_INVERSE));
    cufftDestroy( _fft_dummy );
    //GpuCpuData<float> d = *_waveform.waveform_data;
    fdata = ut.getCpuMemory();
    printf("num_frames/2=%d\n", num_frames/2);
    for (unsigned i=0; i<num_frames; i++)
        printf("\t%g  \t%g\n",fdata[2*i], fdata[2*i+1]);
#endif
}


    /**
      Writes wave audio with 16 bits per sample
      */
void Audiofile::writeFile( const char* filename )
{
	TaskTimer tt("%s %s",__FUNCTION__,filename);

    _last_filename = filename;
    // todo: this method only writes mono data from the first (left) channel

    const int format=SF_FORMAT_WAV | SF_FORMAT_PCM_16;
    //const int format=SF_FORMAT_WAV | SF_FORMAT_FLOAT;

    //int number_of_channels = 1;
    SndfileHandle outfile(filename, SFM_WRITE, format, 1, sample_rate());

    if (!outfile) return;

    outfile.write( _waveform->waveform_data->getCpuMemory(), _waveform->waveform_data->getNumberOfElements().width); // yes float
    //play();
}

pBuffer Audiofile::read( unsigned firstSample, unsigned numberOfSamples ) {
    return  getChunk( firstSample, numberOfSamples, 0, Buffer::Only_Real );
}

/* returns a chunk with numberOfSamples samples. If the requested range exceeds the source signal it is padded with 0. */
pBuffer Audiofile::getChunk( unsigned firstSample, unsigned numberOfSamples, unsigned channel, Buffer::Interleaved interleaved )
{
    if (firstSample+numberOfSamples < firstSample)
        throw std::invalid_argument("Overflow: firstSample+numberOfSamples");

    if (channel >= _waveform->waveform_data->getNumberOfElements().height)
        throw std::invalid_argument("channel >= _waveform.waveform_data->getNumberOfElements().height");

    char m=1+(Buffer::Interleaved_Complex == interleaved);
    char sourcem = 1+(Buffer::Interleaved_Complex == _waveform->interleaved());

    pBuffer chunk( new Buffer( interleaved ));
    chunk->waveform_data.reset( new GpuCpuData<float>(0, make_cudaExtent(m*numberOfSamples, 1, 1) ) );
    chunk->sample_rate = sample_rate();
    chunk->sample_offset = firstSample;
    size_t sourceSamples = _waveform->waveform_data->getNumberOfElements().width/sourcem;

    unsigned validSamples;
    if (firstSample > sourceSamples)
        validSamples = 0;
    else if ( firstSample + numberOfSamples > sourceSamples )
        validSamples = sourceSamples - firstSample;
    else // default case
        validSamples = numberOfSamples;

    float *target = chunk->waveform_data->getCpuMemory();
    float *source = _waveform->waveform_data->getCpuMemory()
                  + firstSample*sourcem
                  + channel * _waveform->waveform_data->getNumberOfElements().width;

    bool interleavedSource = Buffer::Interleaved_Complex == _waveform->interleaved();
    for (unsigned i=0; i<validSamples; i++) {
        target[i*m + 0] = source[i*sourcem + 0];
        if (Buffer::Interleaved_Complex == interleaved)
            target[i*m + 1] = interleavedSource ? source[i*sourcem + 1]:0;
    }

    for (unsigned i=validSamples; i<numberOfSamples; i++) {
        target[i*m + 0] = 0;
        if (Buffer::Interleaved_Complex == interleaved)
            target[i*m + 1] = 0;
    }
    return chunk;
}

    
class SoundPlayer {

public:
    SoundPlayer() {
#ifdef SoundPlayer_VERBOSE
        std::vector< AudioDeviceDesc > devices;
        GetSupportedAudioDevices( devices );
        fprintf(stdout, "%lu audio device%s available:\n", devices.size(), devices.size()==1?"":"s");
        for (unsigned i=0; i<devices.size(); i++) {
            fprintf(stdout, "  %s: %s\n", devices[i].name.c_str(), devices[i].description.c_str());
        }

        _device = OpenDevice();
        if (0==_device.get())
            fprintf(stdout,"Couldn't open sound device\n");
        else
            fprintf(stdout,"Opened sound device: %s\n", _device->getName());

        fflush(stdout);
#endif // SoundPlayer_VERBOSE
        toggle = 0;
    }

    void play( SampleBufferPtr sampleBuffer, float length )
    {
        if (0 == _device.get())
            _device = OpenDevice();
        /*if (0 == _device.get() || !active()) {
            _device = 0;
            _device = OpenDevice();
        }*/

        _length = length;

        _sound[toggle] = OpenSound(_device, sampleBuffer->openStream(), false);
        if (_sound[toggle].get())
            _sound[toggle]->play();
        else {
            fprintf(stderr,"Can't play sound\n");
            fflush(stdout);
        }

        unsigned n = (sizeof(_sound)/sizeof(_sound[0]));

        for(unsigned i=0; i<n; i++)
        {
            if (_sound[i].get()) {
                _sound[i]->setVolume( 1- ((toggle + n - i)%n)/(float)n );
            }
        }

        toggle = (toggle+1)%n;
    }

    bool active() {
        unsigned n = (sizeof(_sound)/sizeof(_sound[0]));
        for(unsigned i=0; i<n; i++)
        {
            if (_sound[i].get()) {
                if (_sound[i]->isPlaying())
                    return true;
            }
        }
        return false;
    }
private:
    AudioDevicePtr _device;
    OutputStreamPtr _sound[10];
    int toggle;
    float _length;
};

pSource Audiofile::crop() {
    // create signed short representation
    unsigned num_frames = _waveform->waveform_data->getNumberOfElements().width;
    unsigned channel_count = _waveform->waveform_data->getNumberOfElements().height;
    float *fdata = _waveform->waveform_data->getCpuMemory();
    unsigned firstNonzero = 0;
    unsigned lastNonzero = 0;
    for (unsigned f=0; f<num_frames; f++)
        for (unsigned c=0; c<channel_count; c++)
            if (fdata[f*channel_count + c])
                lastNonzero = f;
            else if (firstNonzero==f)
                firstNonzero = f+1;

    if (firstNonzero > lastNonzero)
        return pSource();

    Audiofile* wf(new Audiofile());
    pSource rwf(wf);
    wf->_waveform->sample_rate = sample_rate();
    wf->_waveform->waveform_data.reset (new GpuCpuData<float>(0, make_cudaExtent((lastNonzero-firstNonzero+1) , channel_count, 1)));
    float *data = wf->_waveform->waveform_data->getCpuMemory();


    for (unsigned f=firstNonzero; f<=lastNonzero; f++)
        for (unsigned c=0; c<channel_count; c++) {
            float rampup = min(1.f, (f-firstNonzero)/(sample_rate()*0.01f));
            float rampdown = min(1.f, (lastNonzero-f)/(sample_rate()*0.01f));
            rampup = 3*rampup*rampup-2*rampup*rampup*rampup;
            rampdown = 3*rampdown*rampdown-2*rampdown*rampdown*rampdown;
            data[f-firstNonzero + c*num_frames] = 0.5f*rampup*rampdown*fdata[ f + c*num_frames];
        }

    return rwf;
}

void Audiofile::play() {
    pSource wf = this->crop();
    if (!wf)
      return;
    wf->writeFile("selection.wav");
#ifdef __APPLE__
    QSound::play("selection.wav");
    printf("Play file: %s\n", "selection.wav");
    return;
#endif

    Audiofile* wf = dynamic_cast<Audiofile*>(wfs.get());
    wf->writeFile("selection.wav");
    // create signed short representation
    unsigned num_frames = wf->_waveform->waveform_data->getNumberOfElements().width;
    unsigned channel_count = wf->_waveform->waveform_data->getNumberOfElements().height;
    float *fdata = wf->_waveform->waveform_data->getCpuMemory();


    boost::scoped_array<short> data( new short[num_frames * channel_count] );
    for (unsigned f=0; f<num_frames; f++)
        for (unsigned c=0; c<channel_count; c++)
            data[f*channel_count + c] = fdata[ f + c*num_frames]*32767;

    BOOST_ASSERT( 0 != sample_rate() );

    // play sound
    SampleBufferPtr sampleBuffer( CreateSampleBuffer(
            data.get(),
            num_frames,
            channel_count,
            sample_rate(),
            SF_S16));

    static SoundPlayer sp;
    sp.play( sampleBuffer, num_frames / (float)sample_rate() );
}

/*
class ChunkSource: public Audiere::SampleSource {
protected:
    pWaveform _waveform;
    unsigned _position;

    ~ChunkSource() { }

public:
    ChunkSource( pWaveform waveform )
    :   _waveform(waveform),
        _position(0)
    {
    }

    void getFormat(
            int& channel_count,
            int& sample_rate,
            SampleFormat& sample_format)
    {
        channel_count = 1;
        sample_rate = _waveform->sample_rate();
        sample_format = Audiere::SF_S16;
    }

    int read(int frame_count, void* buffer) {
        // ...
    }

    void reset() { return 0; }
    bool isSeekable() { return false; }

    int getLength() { return _waveform->number_of_samples(); }

    void setPosition(int position) { _position = position; }
    int getPosition() { return _position; }

    bool getRepeat() { return false; }
    void setRepeat(bool) {}

    int getTagCount() { return 0; }
    virtual const char* getTagKey(int i) { return 0; }
    virtual const char* getTagValue(int i) { return 0; }
    virtual const char* getTagType(int i) { return 0; }
};
*/
unsigned Audiofile::sample_rate() {          return _waveform->sample_rate;    }
unsigned Audiofile::number_of_samples() {    return _waveform->waveform_data->getNumberOfElements().width; }

} // namespace Signal
