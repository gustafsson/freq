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
    _first_invalid_sample( 0 ),
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
    TaskTimer tt(TaskTimer::LogSimple, "%s: Putting buffer [%u,%u]", __FUNCTION__, buffer->sample_offset, buffer->sample_offset+buffer->number_of_samples());
    buffer->waveform_data->getCpuMemory(); // Make sure the buffer is moved over to CPU memory

    {
        if (!_data.empty()) if (_data.sample_rate() != buffer->sample_rate) {
            throw std::logic_error(std::string(__FUNCTION__) + " sample rate is different from previous sample rate" );
        }

        _first_invalid_sample = buffer->sample_offset + buffer->number_of_samples();

        if (_data.empty())
            _last_timestamp = _first_timestamp = clock();
        else
            _last_timestamp = clock();

        _data.put( buffer );

        unsigned x = expected_samples_left();
        if (x < buffer->number_of_samples() )
            x = 0;
        else
            x -= buffer->number_of_samples();
        expected_samples_left( x );

        if (streamPlayback) {
            if (streamPlayback->isStopped()) {
                // start over
                streamPlayback->start();
            }
            tt.info("Is playing");
            return;
        }
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

    _data.reset();
    _playback_itr = 0;
    expected_samples_left(0);
}

SamplesIntervalDescriptor Playback::
        getMissingSamples()
{
    if (0 == expected_samples_left())
        return SamplesIntervalDescriptor();

    return SamplesIntervalDescriptor(
            _first_invalid_sample,
            _first_invalid_sample + expected_samples_left()
            );
}

unsigned Playback::
        nAccumulatedSamples()
{
    return _data.number_of_samples();
}

pBuffer Playback::
        first_buffer()
{
    return _data.first_buffer();
}

bool Playback::
        isStopped()
{
    return streamPlayback?!streamPlayback->isActive() || streamPlayback->isStopped():true;
}


bool Playback::
        isUnderfed()
{
    unsigned nAccumulated_samples = nAccumulatedSamples();

    if (1>=_data.size())
        return true;

    clock_t first = _first_timestamp;
    clock_t last = _last_timestamp;


    float accumulation_time = (last-first) / (float)CLOCKS_PER_SEC;
    float incoming_samples_per_sec = (nAccumulated_samples - _data.first_buffer()->number_of_samples()) / accumulation_time;
    float time_left = (nAccumulated_samples - _playback_itr + expected_samples_left()) / (float)_data.sample_rate();
    float estimated_time_required = expected_samples_left() / incoming_samples_per_sec;

    if (expected_samples_left() == 0 || time_left > estimated_time_required )
    {
        return false; // not underfed, ok to start playing
    }
    return true; // underfed, don't start playing
}

void Playback::
        preparePlayback( unsigned firstSample, unsigned number_of_samples )
{
    _first_invalid_sample = firstSample;
    expected_samples_left( number_of_samples );
    _playback_itr = firstSample;
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

    TaskTimer tt("Reading %u, %u", _playback_itr, framesPerBuffer );
    pBuffer b = _data.readFixedLength( _playback_itr, framesPerBuffer );
    memcpy( buffer, b->waveform_data->getCpuMemory(), framesPerBuffer*sizeof(float) );
    _playback_itr += framesPerBuffer;

    if (_data.first_buffer()->sample_offset + _data.number_of_samples() + 10*framesPerBuffer < _playback_itr ) {
        tt.info("Done, %u", _data.number_of_samples());
        return paComplete;
    }

    return paContinue;
}

} // namespace Signal
