#include "signal/microphonerecorder.h"
#include <iostream>
#include <memory.h>
#include <boost/foreach.hpp>

using namespace std;

namespace Signal {

MicrophoneRecorder::MicrophoneRecorder(int inputDevice)
{
    portaudio::System &sys = portaudio::System::instance();

    if (0>inputDevice || inputDevice>sys.deviceCount()) {
        inputDevice = sys.defaultInputDevice().index();
    } else if ( sys.deviceByIndex(inputDevice).isOutputOnlyDevice() ) {
        cout << "Requested device '" << sys.deviceByIndex(inputDevice).name() << "' can only be used for output." << endl;
        inputDevice = sys.defaultInputDevice().index();
    } else {
        inputDevice = inputDevice;
    }

    cout << "Using device '" << sys.deviceByIndex(inputDevice).name() << "' for audio input." << endl << endl;

    portaudio::Device& device = sys.deviceByIndex(inputDevice);

    cout << "Opening recording input stream on " << device.name() << endl;
    portaudio::DirectionSpecificStreamParameters inParamsRecord(
            device,
            1, // channels
            portaudio::FLOAT32,
            false, // interleaved
#ifdef __APPLE__
            device.defaultHighInputLatency(),
#else
            device.defaultLowInputLatency(),
#endif
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

    if (0<_data.length()) {
        TaskTimer tt(TaskTimer::LogVerbose, "Releasing recorded data");
        _data.reset();
    }
}

void MicrophoneRecorder::startRecording()
{
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
        read( const Interval& I )
{
    return _data.readFixedLength( I );
}

unsigned MicrophoneRecorder::
        sample_rate()
{
    return _stream_record->sampleRate();
}

long unsigned MicrophoneRecorder::
        number_of_samples()
{
    return _data.number_of_samples();
}

int MicrophoneRecorder::
        writeBuffer(const void *inputBuffer,
                 void * /*outputBuffer*/,
                 unsigned long framesPerBuffer,
                 const PaStreamCallbackTimeInfo * /*timeInfo*/,
                 PaStreamCallbackFlags /*statusFlags*/)
{
    BOOST_ASSERT( inputBuffer );
    const float **in = (const float **)inputBuffer;
    const float *buffer = in[0];

    pBuffer b( new Buffer(0, framesPerBuffer, sample_rate() ) );
    memcpy ( b->waveform_data()->getCpuMemory(), buffer, framesPerBuffer*sizeof(float) );

    b->sample_offset = number_of_samples();
    b->sample_rate = sample_rate();

    _data.put( b );

    // todo put to a postsink as well
    // data_available(this);

    return paContinue;
}

} // namespace Waveform
