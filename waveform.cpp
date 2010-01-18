#include "waveform.h"
#include <audiere.h> // for reading various formats
#include <sndfile.hh> // for writing wav
#include <math.h>
#include "Statistics.h"
#include <stdexcept>
#include <iostream>

using namespace std;
using namespace audiere;


Waveform::Waveform()
:   _source(0),
    _sample_rate(0)
{}


/**
  Reads an audio file using libaudiere
  */
Waveform::Waveform (const char* filename)
{
    _source = OpenSampleSource (filename); // , FileFormat file_format=FF_AUTODETECT
    if (0==_source)
        throw std::ios_base::failure(string() + "File " + filename + " not found");

    SampleFormat sample_format;
    int channel_count, sample_rate;
    _source->getFormat( channel_count, sample_rate, sample_format);
    _sample_rate=sample_rate;

    unsigned frame_size = GetSampleSize(sample_format);
    unsigned num_frames = _source->getLength();

    _waveform.waveform_data.reset( new GpuCpuData<float>(0, make_uint3( num_frames, channel_count, 1)) );
    std::vector<char> data(num_frames*frame_size*channel_count);
    _source->read(num_frames, data.data());

    unsigned j=0;
    float* fdata = _waveform.waveform_data->getCpuMemory();

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

    Statistics<float> waveform( _waveform.waveform_data.get() );
}


    /**
      Writes wave audio with 16 bits per sample
      */
void Waveform::writeFile( const char* filename ) const
{
    // todo: this method only writes mono data from the first (left) channel

    const int format=SF_FORMAT_WAV | SF_FORMAT_PCM_16;
    //  const int format=SF_FORMAT_WAV | SF_FORMAT_FLOAT;

    //int number_of_channels = 1;
    SndfileHandle outfile(filename, SFM_WRITE, format, 1, _sample_rate);

    if (not outfile) return;

    outfile.write( _waveform.waveform_data->getCpuMemory(), _waveform.waveform_data->getNumberOfElements().width); // yes float
}


/* returns a chunk with numberOfSamples samples. If the requested range exceeds the source signal it is padded with 0. */
pWaveform_chunk Waveform::getChunk( unsigned firstSample, unsigned numberOfSamples, unsigned channel, Waveform_chunk::Interleaved interleaved )
{
    if (firstSample+numberOfSamples < firstSample)
        throw std::invalid_argument("Overflow: firstSample+numberOfSamples");

    if (channel >= _waveform.waveform_data->getNumberOfElements().height)
        throw std::invalid_argument("channel >= _waveform.waveform_data->getNumberOfElements().height");

    char m=1+(Waveform_chunk::Interleaved_Complex == interleaved);
    char sourcem = 1+(Waveform_chunk::Interleaved_Complex == _waveform.interleaved());

    pWaveform_chunk chunk( new Waveform_chunk( interleaved ));
    chunk->waveform_data.reset( new GpuCpuData<float>(0, make_cudaExtent(m*numberOfSamples, 1, 1) ) );
    chunk->sample_rate = _sample_rate;
    chunk->sample_offset = firstSample;
    size_t sourceSamples = _waveform.waveform_data->getNumberOfElements().width/sourcem;

    unsigned validSamples;
    if (firstSample > sourceSamples)
        validSamples = 0;
    else if ( firstSample + numberOfSamples > sourceSamples )
        validSamples = numberOfSamples - firstSample;
    else // default case
        validSamples = numberOfSamples;

    float *target = chunk->waveform_data->getCpuMemory();
    float *source = _waveform.waveform_data->getCpuMemory()
                  + firstSample*sourcem
                  + channel * _waveform.waveform_data->getNumberOfElements().width;

    bool interleavedSource = Waveform_chunk::Interleaved_Complex == _waveform.interleaved();
    for (unsigned i=0; i<validSamples; i++) {
        target[i*m + 0] = source[i*m + 0];
        if (Waveform_chunk::Interleaved_Complex == interleaved)
            target[i*m + 1] = interleavedSource ? source[i*m + 1]:0;
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
