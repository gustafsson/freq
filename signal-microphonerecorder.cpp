#include "signal-microphonerecorder.h"
#include <iostream>
#include <memory.h>
#include <boost/foreach.hpp>

using namespace std;

namespace Signal {

MicrophoneRecorder::MicrophoneRecorder(int inputDevice)
:   _callback(0)
{
    portaudio::System &sys = portaudio::System::instance();

    if (0>inputDevice || inputDevice>sys.deviceCount()) {
        inputDevice = sys.defaultOutputDevice().index();
    } else if ( sys.deviceByIndex(inputDevice).isOutputOnlyDevice() ) {
        cout << "Requested device '" << sys.deviceByIndex(inputDevice).name() << "' can only be used for output." << endl;
        inputDevice = sys.defaultOutputDevice().index();
    } else {
        inputDevice = inputDevice;
    }

    cout << "Using device '" << sys.deviceByIndex(inputDevice).name() << "' for input." << endl << endl;

    portaudio::Device& device = sys.deviceByIndex(inputDevice);

    cout << "Opening recording input stream on " << device.name() << endl;
    portaudio::DirectionSpecificStreamParameters inParamsRecord(
            device,
            1, // channels
            portaudio::FLOAT32,
            false, // interleaved
            device.defaultLowInputLatency(),
            NULL);

    portaudio::StreamParameters paramsRecord(
            inParamsRecord,
            portaudio::DirectionSpecificStreamParameters::null(),
            device.defaultSampleRate(),
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
        _stream_record->isStopped()? void(): _stream_record->abort();
        _stream_record->close();
    }

    if (!_cache.empty()) {
        TaskTimer tt(TaskTimer::LogVerbose, "Releasing recorded data");
        _cache.clear();
    }
}

void MicrophoneRecorder::startRecording( Signal::Sink* p )
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
    // TODO refactor and use a BufferSource instead, shared with Playback::readBuffer
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

    _cache.push_back( b );

    if (_callback)
        _callback->put( b, this );

    return paContinue;
}

} // namespace Waveform
