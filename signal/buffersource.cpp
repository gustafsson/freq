#ifdef _MSC_VER
typedef __int64 __int64_t;
#else
#include <stdint.h> // defines __int64_t which is expected by sndfile.h
#endif

#include "signal/buffersource.h"
#include "signal/playback.h"
#include <sndfile.hh> // for reading various formats
#include <math.h>
#include "Statistics.h"
#include <stdexcept>
#include <iostream>
#include <sstream>

#include <boost/scoped_array.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/algorithm/string.hpp>
//#include <QThread>
//#include <QSound>

#if LEKA_FFT
#include <cufft.h>

#endif

using namespace std;

namespace Signal {


BufferSource::
        BufferSource( pBuffer waveform )
:   read_channel(0),
    _waveform(waveform)
{
}


pBuffer BufferSource::
        read( Interval I )
{
    return  getChunk( I.first, I.count, read_channel, Buffer::Only_Real );
}

// TODO rewrite getChunk and use only 'read'. Buffer has functions for transfering between real and complex data
/* returns a chunk with numberOfSamples samples. If the requested range exceeds the source signal it is padded with 0. */
pBuffer BufferSource::
        getChunk( unsigned firstSample, unsigned numberOfSamples, unsigned channel, Buffer::Interleaved interleaved )
{
    // validate arguments
    if (firstSample+numberOfSamples < firstSample)
        throw std::invalid_argument("Overflow: firstSample+numberOfSamples");

    if (channel >= _waveform->waveform_data->getNumberOfElements().height)
        throw std::invalid_argument("channel >= _waveform.waveform_data->getNumberOfElements().height");

    char m = Buffer::Interleaved_Complex == interleaved ? 2:1;
    char sourcem = Buffer::Interleaved_Complex == _waveform->interleaved() ?2:1;

    pBuffer chunk( new Buffer( interleaved ));
    chunk->waveform_data.reset( new GpuCpuData<float>(0, make_cudaExtent(m*numberOfSamples, 1, 1) ) );
    chunk->sample_rate = sample_rate();
    chunk->sample_offset = firstSample;
    size_t sourceSamples = _waveform->number_of_samples();

    unsigned leading_zeros = 0;
    if(firstSample > _waveform->sample_offset)
        firstSample -= _waveform->sample_offset;
    else
    {
        leading_zeros = _waveform->sample_offset - firstSample;
        firstSample = 0;
    }

    float *target = chunk->waveform_data->getCpuMemory();
    float *source = _waveform->waveform_data->getCpuMemory()
                  + firstSample*sourcem
                  + channel * _waveform->waveform_data->getNumberOfElements().width;

    if (firstSample > sourceSamples)
        validSamples = 0;
    else if ( firstSample + numberOfSamples > sourceSamples )
        validSamples = sourceSamples - firstSample;
    else // default case
        validSamples = numberOfSamples;


    for (unsigned i=0; i<leading_zeros; i++) {
        target[i*m + 0] = 0;
        if (Buffer::Interleaved_Complex == interleaved)
            target[i*m + 1] = 0;
    }
    target = &target[leading_zeros*m];
    numberOfSamples -= leading_zeros;
    validSamples -= leading_zeros;


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


/**
  Remove zeros from the beginning and end
  */
pSource BufferSource::
        crop()
{
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

    BufferSource* wf(new BufferSource());
    pSource rwf(wf);
    wf->_waveform->sample_offset = firstNonzero + _waveform->sample_offset;
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

} // namespace Signal
