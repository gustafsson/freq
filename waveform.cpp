#include "waveform.h"
#include <audiere.h> // for reading various formats
#include <sndfile.hh> // for writing wav
#include <math.h>
#include "Statistics.h"
#include <stdexcept>
using namespace audiere;

#include <iostream>
using namespace std;

Waveform::Waveform()
:   _sample_rate(0),
    _source(0)
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

    /**
      Writes wave audio with 16 bits per sample
      */
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

pWaveform_chunk Waveform::getChunk( unsigned firstSample, unsigned numberOfSamples, int channel=0, bool interleaved=true )
{
    pWaveform_chunk chunk( new Waveform_chunk( Waveform_chunk::Interleaved_Complex ));
    char m=1+interleaved;
    chunk->data.reset( new GpuCpuData<float>(0, makeCudaExtent(m*numberOfSamples, 1, 1) ) );

    float *original = _waveform->data->getCpuMemory() + firstSample;
    float *complex = chunk->data->getCpuMemory();
    _waveform->data->getNumberOfSamples().width < fistSample+numberOfSamples;
    unsigned validSamples = numberOfSampels;
    if (firstSample > _waveform->data->getNumberOfSamples().width)
        validSamples = 0;
    else if ( numberOfSampels > _waveform->data->getNumberOfSamples().width - firstSample)
        validSamples = _waveform->data->getNumberOfSamples().width - firstSample;
    for (int i=0; i<validSamples; i++) {
        complex[i*m + 0] = original[i];
        if (interleaved)
            complex[i*m + 1] = 0;
    }
    for (int i=validSamples; i<numberOfSampels; i++) {
        complex[i*m + 0] = 0;
        if (interleaved)
            complex[i*m + 1] = 0;
    }
    return chunk;
}
    
void Waveform_chunk::interleaved(Interleaved value) {
    if (value == _interleaved)
        return;

    boost::scoped_ptr<GpuCpuData<float> > data;
    chunk->data.reset( new GpuCpuData<float>(0, makeCudaExtent(m*numberOfSamples, 1, 1) ) );

}

Waveform_chunk::Waveform_chunk(Interleaved interleaved)
:   interleaved(interleaved)
{
    switch(interleaved) {
        case Interleaved_Complex:
        case Only_Real:
            break;
        default:
            throw invalid_argument( string(__FUNCTION___));
    }
}

pWaveform_chunk Waveform_chunk::getInterleaved(Interleaved value)
{
    pWaveform_chunk chunk( new Waveform_chunk( Waveform_chunk::Interleaved_Complex ));

    if (value == _interleaved) {
        chunk->data.reset( new GpuCpuData<float>(data->getCpuMemory(), data->getNumberOfElements() ) );
        return chunk;
    }

    cudaExtent orgSz = data->getNumberOfElements();

    //makeCudaExtent(m*numberOfSamples, 1, 1)
    if (value == Only_Real) {
        cudaExtent realSz = orgSz;
        realSz.width/=2;
        chunk->data.reset( new GpuCpuData<float>(0, realSz ) );

        float *complex = data->getCpuMemory();
        float *real = chunk->data->getCpuMemory();

        for (int z=0; z<realSz.depth; z++)
            for (int y=0; y<realSz.height; y++)
                for (int x=0; x<szrealSzwidth; x++)
                    real[ x + (y + z*realSz.height)*realSz.width ]
                            = complex[ 2*x + (y + z*orgSz.height)*orgSz.width ];
    } else if (value == Interleaved_Complex) {
        cudaExtent complexSz = orgSz;
        complexSz.width*=2;
        chunk->data.reset( new GpuCpuData<float>(0, complexSz ) );

        float *complex = chunk->data->getCpuMemory();
        float *real = data->getCpuMemory();

        for (int z=0; z<orgSz.depth; z++)
            for (int y=0; y<orgSz.height; y++)
                for (int x=0; x<orgSz.width; x++)
                {
                    complex[ 2*x + (y + z*complex.height)*complex.width ]
                            = real[ x + (y + z*orgSz.height)*orgSz.width ];
                    complex[ 2*x + 1 + (y + z*complex.height)*complex.width ] = 0;
                }
    } else {
        throw invalid_argument( string(__FUNCTION___));
    }

    return chunk;
}

pWaveform_chunk ::interleaved( unsigned firstSample, unsigned numberOfSamples, int channel=0, bool interleaved=true )
{
    pWaveform_chunk chunk( new Waveform_chunk( Waveform_chunk::Interleaved_Complex ));
    char m=1+interleaved;
    chunk->data.reset( new GpuCpuData<float>(0, makeCudaExtent(m*numberOfSamples, 1, 1) ) );

    float *original = _waveform->data->getCpuMemory() + firstSample;
    float *complex = chunk->data->getCpuMemory();
    _waveform->data->getNumberOfSamples().width < fistSample+numberOfSamples;
    unsigned validSamples = numberOfSampels;
    if (firstSample > _waveform->data->getNumberOfSamples().width)
        validSamples = 0;
    else if ( numberOfSampels > _waveform->data->getNumberOfSamples().width - firstSample)
        validSamples = _waveform->data->getNumberOfSamples().width - firstSample;
    for (int i=0; i<validSamples; i++) {
        complex[i*m + 0] = original[i];
        if (interleaved)
            complex[i*m + 1] = 0;
    }
    for (int i=validSamples; i<numberOfSampels; i++) {
        complex[i*m + 0] = 0;
        if (interleaved)
            complex[i*m + 1] = 0;
    }
    return chunk;
}
