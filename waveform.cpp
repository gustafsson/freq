#include "waveform.h"
#include <audiere.h> // for reading various formats
#include <sndfile.hh> // for writing wav
#include <math.h>
#include "Statistics.h"
#include <stdexcept>
#include <iostream>

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


Waveform::Waveform()
:   _source(0),
    _sample_rate(0)
{
    _waveform.reset( new Waveform_chunk());
}


/**
  Reads an audio file using libaudiere
  */
Waveform::Waveform (const char* filename)
{
    _waveform.reset( new Waveform_chunk());

    _source = OpenSampleSource (filename); // , FileFormat file_format=FF_AUTODETECT
    if (0==_source)
        throw std::ios_base::failure(string() + "File " + filename + " not found\n"
            "\n"
            "Supported audio file formats through Audiere: Ogg Vorbis, MP3, FLAC, Speex, uncompressed WAV, AIFF, MOD, S3M, XM, IT");

    SampleFormat sample_format;
    int channel_count, sample_rate;
    _source->getFormat( channel_count, sample_rate, sample_format);
    _sample_rate=sample_rate;

    unsigned frame_size = GetSampleSize(sample_format);
    unsigned num_frames = _source->getLength();

    if (0==num_frames)
        throw std::ios_base::failure(string() + "Failed reding file " + filename);

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

    _waveform = getChunk( 0, number_of_samples(), 0, Waveform_chunk::Only_Real );
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
void Waveform::writeFile( const char* filename )
{
    _last_filename = filename;
    // todo: this method only writes mono data from the first (left) channel

    const int format=SF_FORMAT_WAV | SF_FORMAT_PCM_16;
    //  const int format=SF_FORMAT_WAV | SF_FORMAT_FLOAT;

    //int number_of_channels = 1;
    SndfileHandle outfile(filename, SFM_WRITE, format, 1, _sample_rate);

    if (!outfile) return;

    outfile.write( _waveform->waveform_data->getCpuMemory(), _waveform->waveform_data->getNumberOfElements().width); // yes float
    //play();
}


/* returns a chunk with numberOfSamples samples. If the requested range exceeds the source signal it is padded with 0. */
pWaveform_chunk Waveform::getChunk( unsigned firstSample, unsigned numberOfSamples, unsigned channel, Waveform_chunk::Interleaved interleaved )
{
    if (firstSample+numberOfSamples < firstSample)
        throw std::invalid_argument("Overflow: firstSample+numberOfSamples");

    if (channel >= _waveform->waveform_data->getNumberOfElements().height)
        throw std::invalid_argument("channel >= _waveform.waveform_data->getNumberOfElements().height");

    char m=1+(Waveform_chunk::Interleaved_Complex == interleaved);
    char sourcem = 1+(Waveform_chunk::Interleaved_Complex == _waveform->interleaved());

    pWaveform_chunk chunk( new Waveform_chunk( interleaved ));
    chunk->waveform_data.reset( new GpuCpuData<float>(0, make_cudaExtent(m*numberOfSamples, 1, 1) ) );
    chunk->sample_rate = _sample_rate;
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

    bool interleavedSource = Waveform_chunk::Interleaved_Complex == _waveform->interleaved();
    for (unsigned i=0; i<validSamples; i++) {
        target[i*m + 0] = source[i*sourcem + 0];
        if (Waveform_chunk::Interleaved_Complex == interleaved)
            target[i*m + 1] = interleavedSource ? source[i*sourcem + 1]:0;
    }

    for (unsigned i=validSamples; i<numberOfSamples; i++) {
        target[i*m + 0] = 0;
        if (Waveform_chunk::Interleaved_Complex == interleaved)
            target[i*m + 1] = 0;
    }
    return chunk;
}

    
Waveform_chunk::Waveform_chunk(Interleaved interleaved)
:   sample_offset(0),
    sample_rate(0),
    modified(0),
    _interleaved(interleaved)
{
    switch(_interleaved) {
        case Interleaved_Complex:
        case Only_Real:
            break;
        default:
            throw invalid_argument( string( __FUNCTION__ ));
    }
}


pWaveform_chunk Waveform_chunk::getInterleaved(Interleaved value)
{
    pWaveform_chunk chunk( new Waveform_chunk( value ));

    if (value == _interleaved) {
        chunk->waveform_data.reset( new GpuCpuData<float>(waveform_data->getCpuMemory(), waveform_data->getNumberOfElements() ) );
        return chunk;
    }

    cudaExtent orgSz = waveform_data->getNumberOfElements();

    //makeCudaExtent(m*numberOfSamples, 1, 1)
    switch(value) {
        case Only_Real: {
            cudaExtent realSz = orgSz;
            realSz.width/=2;
            chunk->waveform_data.reset( new GpuCpuData<float>(0, realSz ) );

            float *complex = waveform_data->getCpuMemory();
            float *real = chunk->waveform_data->getCpuMemory();

            for (unsigned z=0; z<realSz.depth; z++)
                for (unsigned y=0; y<realSz.height; y++)
                    for (unsigned x=0; x<realSz.width; x++)
                        real[ x + (y + z*realSz.height)*realSz.width ]
                                = complex[ 2*x + (y + z*orgSz.height)*orgSz.width ];
            break;
        }
        case Interleaved_Complex: {
            cudaExtent complexSz = orgSz;
            complexSz.width*=2;
            chunk->waveform_data.reset( new GpuCpuData<float>(0, complexSz ) );

            float *complex = chunk->waveform_data->getCpuMemory();
            float *real = waveform_data->getCpuMemory();

            for (unsigned z=0; z<orgSz.depth; z++)
                for (unsigned y=0; y<orgSz.height; y++)
                    for (unsigned x=0; x<orgSz.width; x++)
                    {
                        complex[ 2*x + (y + z*complexSz.height)*complexSz.width ]
                                = real[ x + (y + z*orgSz.height)*orgSz.width ];
                        complex[ 2*x + 1 + (y + z*complexSz.height)*complexSz.width ] = 0;
                    }
            break;
        }
        default:
            throw invalid_argument( string(__FUNCTION__));
    }

    return chunk;
}

class SoundPlayer {

public:
    SoundPlayer() {
        _device = OpenDevice();
        toggle = 0;
    }

    void play( SampleBufferPtr sampleBuffer, float length )
    {
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

private:
    AudioDevicePtr _device;
    OutputStreamPtr _sound[10];
    int toggle;
    float _length;
};

pWaveform Waveform::crop() {
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
        return pWaveform();

    pWaveform wf(new Waveform());
    wf->_sample_rate = _sample_rate;
    wf->_waveform->waveform_data.reset (new GpuCpuData<float>(0, make_cudaExtent((lastNonzero-firstNonzero+1) , channel_count, 1)));
    float *data = wf->_waveform->waveform_data->getCpuMemory();


    for (unsigned f=firstNonzero; f<=lastNonzero; f++)
        for (unsigned c=0; c<channel_count; c++) {
            float rampup = min(1.f, (f-firstNonzero)/(_sample_rate*0.01f));
            float rampdown = min(1.f, (lastNonzero-f)/(_sample_rate*0.01f));
            rampup = 3*rampup*rampup-2*rampup*rampup*rampup;
            rampdown = 3*rampdown*rampdown-2*rampdown*rampdown*rampdown;
            data[f-firstNonzero + c*num_frames] = rampup*rampdown*fdata[ f + c*num_frames];
        }

    return wf;
}

void Waveform::play() {
#ifdef MAC
    QSound::play( _last_filename.c_str() );
    return;
#endif

    pWaveform wf = this->crop();

    if (!wf.get())
        return;

    // create signed short representation
    unsigned num_frames = wf->_waveform->waveform_data->getNumberOfElements().width;
    unsigned channel_count = wf->_waveform->waveform_data->getNumberOfElements().height;
    float *fdata = wf->_waveform->waveform_data->getCpuMemory();


    boost::scoped_array<short> data( new short[num_frames * channel_count] );
    for (unsigned f=0; f<num_frames; f++)
        for (unsigned c=0; c<channel_count; c++)
            data[f*channel_count + c] = fdata[ f + c*num_frames]*32767;


    // play sound
    SampleBufferPtr sampleBuffer( CreateSampleBuffer(
            data.get(),
            num_frames,
            channel_count,
            _sample_rate,
            SF_S16));

    static SoundPlayer sp;
    sp.play( sampleBuffer, num_frames / (float)_sample_rate );
}
