#include "signal-microphonerecorder.h"
#include <iostream>
#include <memory.h>
#include <boost/foreach.hpp>

using namespace std;

namespace Signal {

MicrophoneRecorder::MicrophoneRecorder()
:   _callback(0)
{
    portaudio::System &sys = portaudio::System::instance();
    portaudio::Device& inputDevice = sys.defaultInputDevice();

    cout << "Opening recording input stream on " << inputDevice.name() << endl;
    portaudio::DirectionSpecificStreamParameters inParamsRecord(
            inputDevice,
            1, // channels
            portaudio::FLOAT32,
            false, // interleaved
            inputDevice.defaultLowInputLatency(),
            NULL);

    portaudio::StreamParameters paramsRecord(
            inParamsRecord,
            portaudio::DirectionSpecificStreamParameters::null(),
            inputDevice.defaultSampleRate(),
            paFramesPerBufferUnspecified,
            paNoFlag);

    _stream_record.reset( new portaudio::MemFunCallbackStream<MicrophoneRecorder>(
            paramsRecord,
            *this,
            &MicrophoneRecorder::writeBuffer));
}

MicrophoneRecorder::~MicrophoneRecorder()
{
    if (_stream_record) {
        _stream_record->isStopped()? void(): _stream_record->stop();
        _stream_record->close();
    }
}

void MicrophoneRecorder::startRecording( Callback *p )
{
    _callback = p;
    _stream_record->start();
}

void MicrophoneRecorder::stopRecording()
{
    _stream_record->stop();
}

bool MicrophoneRecorder::isStopped()
{
    return _stream_record->isStopped();
}

pBuffer MicrophoneRecorder::
        read( unsigned firstSample, unsigned numberOfSamples )
{
    pBuffer b( new Buffer() );
    b->waveform_data.reset( new GpuCpuData<float>( 0, make_cudaExtent( numberOfSamples, 1, 1) ) );
    b->sample_offset = firstSample;
    b->sample_rate = this->sample_rate();

    // this code is close to identical to "Playback::readBuffer", they could both use a BufferSource instead.
    // TODO refactor
    float *buffer = b->waveform_data->getCpuMemory();
    unsigned iBuffer = 0;
    unsigned nAccumulated_samples = 0;

    while ( 0 < numberOfSamples )
    {
        // count previous samples
        for (; iBuffer<_cache.size(); iBuffer++) {
            if (firstSample < nAccumulated_samples + _cache[iBuffer]->number_of_samples() )
                break;
            nAccumulated_samples += _cache[iBuffer]->number_of_samples();
        }

        if (iBuffer>=_cache.size()) {
            memset(buffer, 0, numberOfSamples*sizeof(float));
            numberOfSamples = 0;

        } else {
            unsigned nSamples_to_copy = nAccumulated_samples + _cache[iBuffer]->number_of_samples() - firstSample;
            if (numberOfSamples < nSamples_to_copy )
                nSamples_to_copy = numberOfSamples;

            memcpy( buffer,
                    &_cache[iBuffer]->waveform_data->getCpuMemory()[ firstSample - nAccumulated_samples ],
                    nSamples_to_copy*sizeof(float));

            numberOfSamples -= nSamples_to_copy;
            firstSample += nSamples_to_copy;
            buffer += nSamples_to_copy;
        }
    }

    return b;
}

unsigned MicrophoneRecorder::
        sample_rate()
{
    return _stream_record->sampleRate();
}

unsigned MicrophoneRecorder::
        number_of_samples()
{
    unsigned n = 0;

    BOOST_FOREACH( const pBuffer& s, _cache ) {
        n += s->number_of_samples();
    }

    return n;
}

int MicrophoneRecorder::
        writeBuffer(const void *inputBuffer,
                 void */*outputBuffer*/,
                 unsigned long framesPerBuffer,
                 const PaStreamCallbackTimeInfo */*timeInfo*/,
                 PaStreamCallbackFlags /*statusFlags*/)
{
    BOOST_ASSERT( inputBuffer );
    const float **in = (const float **)inputBuffer;
    const float *buffer = in[0];

    pBuffer b( new Buffer() );
    b->waveform_data.reset( new GpuCpuData<float>( 0, make_cudaExtent( framesPerBuffer, 1, 1) ) );


    memcpy ( b->waveform_data->getCpuMemory(), buffer, framesPerBuffer*sizeof(float) );
    //memset ( b->waveform_data->getCpuMemory(), 0, framesPerBuffer*sizeof(float) );

    _cache.push_back( b );

    if (_callback)
        _callback->recievedData( this );

    return paContinue;
}

} // namespace Waveform
