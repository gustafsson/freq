#include "signal-playback.h"
#include <iostream>
#include <stdexcept>
#include <boost/foreach.hpp>
#include <QMutexLocker>
#include <stdio.h> // todo remove

using namespace std;

namespace Signal {

Playback::
        Playback( int outputDevice )
:   _playback_itr(0),
    _output_device(0)
{
    portaudio::AutoSystem autoSys;
    portaudio::System &sys = portaudio::System::instance();

    static bool first = true;
    if (first) list_devices();

    if (0>outputDevice || outputDevice>sys.deviceCount()) {
        _output_device = sys.defaultOutputDevice().index();
    } else if ( sys.deviceByIndex(outputDevice).isInputOnlyDevice() ) {
        if(first) cout << "Requested device '" << sys.deviceByIndex(outputDevice).name() << "' can only be used for input." << endl;
        _output_device = sys.defaultOutputDevice().index();
    } else {
        _output_device = outputDevice;
    }

    if(first) cout << "Using device '" << sys.deviceByIndex(_output_device).name() << "' for output." << endl << endl;

    //first = false;
}

Playback::
        ~Playback()
{
    cout << "Clearing Playback" << endl;
    if (streamPlayback) {
        streamPlayback->stop();
        streamPlayback->close();
    }
}

/*static*/ void Playback::
        list_devices()
{
    portaudio::AutoSystem autoSys;
    portaudio::System &sys = portaudio::System::instance();

    int 	iNumDevices 		= sys.deviceCount();
    int 	iIndex 				= 0;
    string	strDetails			= "";

    std::cout << "Enumerating sound devices (count " << iNumDevices << ")" << std::endl;
    for (portaudio::System::DeviceIterator i = sys.devicesBegin(); i != sys.devicesEnd(); ++i)
    {
        strDetails = "";
        if ((*i).isSystemDefaultInputDevice())
                strDetails += ", default input";
        if ((*i).isSystemDefaultOutputDevice())
                strDetails += ", default output";

        cout << " ";
        cout << (*i).index() << ": " << (*i).name() << ", ";
        cout << "in=" << (*i).maxInputChannels() << " ";
        cout << "out=" << (*i).maxOutputChannels() << ", ";
        cout << (*i).hostApi().name();

        cout << strDetails.c_str() << endl;

        iIndex++;
    }
}

unsigned Playback::
        playback_itr()
{
    return _playback_itr;
}

float Playback::
        time()
{
    return streamPlayback?streamPlayback->time():0;
}

float Playback::
        outputLatency()
{
    return streamPlayback?streamPlayback->outputLatency():0;
}

void Playback::
        put( pBuffer buffer )
{
    TaskTimer tt("Playback::put [%u,%u]", buffer->sample_offset, buffer->sample_offset+buffer->number_of_samples());
    //TaskTimer tt(TaskTimer::LogVerbose, "%s: Putting buffer [%u,%u]", __FUNCTION__, buffer->sample_offset, buffer->sample_offset+buffer->number_of_samples());

    _last_timestamp = clock();
    if (SinkSource::empty())
    {
        _first_timestamp = _last_timestamp;
        _playback_itr = buffer->sample_offset;
        tt.info("Setting _playback_itr=%u", _playback_itr);
    }

    SinkSource::put( buffer );

    // Make sure the buffer is moved over to CPU memory.
    // (because the audio stream callpack is executed from a different thread it can't
    // access the GPU memory)
    buffer->waveform_data->getCpuMemory();
    // TODO Should perhaps relase GPU memory as well...

    if (streamPlayback)
    {
        if (streamPlayback->isStopped()) {
            // start over
            streamPlayback->start();
        }
        tt.info("Is playing");
        return;
    }

    if (isUnderfed() ) {
        tt.info("Waiting for more data");
        //  Wait for more data

    } else {
        portaudio::System &sys = portaudio::System::instance();

        tt.info("Start playing on: %s", sys.deviceByIndex(_output_device).name() );

        // Set up the parameters required to open a (Callback)Stream:
        portaudio::DirectionSpecificStreamParameters outParamsPlayback(
                sys.deviceByIndex(_output_device),
                1, // mono sound
                portaudio::FLOAT32,
                false,
                sys.deviceByIndex(_output_device).defaultLowOutputLatency(),
                //sys.deviceByIndex(_output_device).defaultHighOutputLatency(),
                NULL);
        portaudio::StreamParameters paramsPlayback(
                portaudio::DirectionSpecificStreamParameters::null(),
                outParamsPlayback,
                buffer->sample_rate,
                0,
                paNoFlag);//paClipOff);

        // Create (and open) a new Stream:
        streamPlayback.reset( new portaudio::MemFunCallbackStream<Playback>(
                paramsPlayback,
                *this,
                &Signal::Playback::readBuffer) );

        if (streamPlayback)
            streamPlayback->start();
    }
}

void Playback::
        reset()
{
    if (streamPlayback)
    {
        // streamPlayback->stop will invoke a join with readBuffer
        if (!streamPlayback->isStopped())
            streamPlayback->stop();
    }

    SinkSource::reset();
    _playback_itr = 0;
}

bool Playback::
        finished()
{
    return expected_samples().isEmpty() && isStopped();
}

bool Playback::
        isStopped()
{
    return streamPlayback?!streamPlayback->isActive() || streamPlayback->isStopped():true;
}

bool Playback::
        isUnderfed()
{
    unsigned nAccumulated_samples = SinkSource::number_of_samples();

    if (10>=SinkSource::size())
        return true; // Haven't received much data, wait to do a better estimate

    if (_expected_samples.isEmpty())
        return false; // No more expected samples, not underfed


    float accumulation_time = (_last_timestamp - _first_timestamp) / (float)CLOCKS_PER_SEC;

    // _first_timestamp is taken after the first buffer,
    // _last_timestamp is taken after the last buffer,
    // that means that accumulation_time is the time it took to accumulate all buffers except
    // the first buffer.
    float incoming_samples_per_sec = (nAccumulated_samples - SinkSource::first_buffer()->number_of_samples()) / accumulation_time;

    float time_left =
            (_expected_samples.intervals().back().last
             - _playback_itr) / (float)SinkSource::sample_rate();

    // Add small margin
    time_left += .05f;

    float estimated_time_required =
            (_expected_samples.intervals().back().last
             - _expected_samples.intervals().front().first) / incoming_samples_per_sec;

    // Return if the estimated time to receive all expected samples is greater than
    // the time it would take to play the remaining part of the data.
    // If it is, the sink is underfed.
    return time_left < estimated_time_required;
}

int Playback::
        readBuffer(const void * /*inputBuffer*/,
                 void *outputBuffer,
                 unsigned long framesPerBuffer,
                 const PaStreamCallbackTimeInfo * /*timeInfo*/,
                 PaStreamCallbackFlags /*statusFlags*/)
{
    BOOST_ASSERT( outputBuffer );
    float **out = static_cast<float **>(outputBuffer);
    float *buffer = out[0];

//    TaskTimer tt(TaskTimer::LogVerbose, "Reading %u, %u", _playback_itr, framesPerBuffer );
    TaskTimer tt("Reading %u, %u", _playback_itr, framesPerBuffer );
    if (SinkSource::first_buffer()->sample_offset + SinkSource::number_of_samples() < _playback_itr + framesPerBuffer) {
        tt.info("Reading past end");
    }
    pBuffer b = SinkSource::readFixedLength( _playback_itr, framesPerBuffer );
    memcpy( buffer, b->waveform_data->getCpuMemory(), framesPerBuffer*sizeof(float) );
    _playback_itr += framesPerBuffer;

    if (SinkSource::first_buffer()->sample_offset + SinkSource::number_of_samples() + 10*2024/*framesPerBuffer*/ < _playback_itr ) {
        tt.info("Reading done, %u", SinkSource::number_of_samples());
        return paComplete;
    }

    return paContinue;
}

} // namespace Signal
