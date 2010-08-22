#include "playback.h"

#include <iostream>
#include <stdexcept>
#include <boost/foreach.hpp>
#include <QMutexLocker>
#include <stdio.h> // todo remove
#include <QMessageBox>

//#define TIME_PLAYBACK
#define TIME_PLAYBACK if(0)

using namespace std;
using namespace boost::posix_time;

namespace Signal {

Playback::
        Playback( int outputDevice )
:   _data( SinkSource::AcceptStrategy_ACCEPT_EXPECTED_ONLY ),
	_first_buffer_size(0),
    _playback_itr(0),
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

    // first = false;
}

Playback::
        ~Playback()
{
    cout << "Clearing Playback" << endl;
    if (streamPlayback) {
        if (!streamPlayback->isStopped())
            streamPlayback->stop();
        if (streamPlayback->isOpen())
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

    cout << "Enumerating sound devices (count " << iNumDevices << ")" << endl;
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
    if(_data.empty())
        return 0.f;

    // streamPlayback->time() doesn't seem to work (ubuntu 10.04)
    // float dt = streamPlayback?streamPlayback->time():0;
    
    time_duration d = microsec_clock::local_time() - _startPlay_timestamp;
    float dt = d.total_milliseconds()*0.001f;
    float t = dt;
    t += _data.first_buffer()->sample_offset / (float)sample_rate();
    t -= 0.08f;
#ifdef _WIN32
    t -= outputLatency();
#endif
    return std::max(0.f, t);
}

float Playback::
        outputLatency()
{
    return streamPlayback?streamPlayback->outputLatency():0;
}

void Playback::
        put( pBuffer buffer )
{
    TIME_PLAYBACK TaskTimer tt("Playback::put [%u,%u]", buffer->sample_offset, buffer->sample_offset+buffer->number_of_samples());

    _last_timestamp = microsec_clock::local_time();
    if (_data.empty())
	{
        _first_timestamp = _last_timestamp;
		_first_buffer_size = buffer->number_of_samples();
	}
    _data.put( buffer );

    // Make sure the buffer is moved over to CPU memory.
    // (because the audio stream callback is executed from a different thread
    // it can't access the GPU memory)
    buffer->waveform_data->getCpuMemory();
    // TODO Should perhaps relase GPU memory as well...

    if (streamPlayback)
    {
        if (streamPlayback->isStopped()) {
            // start over
            streamPlayback->start();
        }
        TIME_PLAYBACK TaskTimer("Is playing").suppressTiming();
        return;
    }

    onFinished();
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
}

bool Playback::
        isFinished()
{
    return expected_samples().isEmpty() && isStopped();
}

void Playback::
        onFinished()
{
    if (isUnderfed() )
    {
        TIME_PLAYBACK TaskTimer("Waiting for more data");
        return;
    }

    try
    {

    // Be nice, don't just destroy the previous one but ask it to stop first
    if (streamPlayback && !streamPlayback->isStopped())
    {
        streamPlayback->stop();
    }
    streamPlayback.reset();

    portaudio::System &sys = portaudio::System::instance();

    TIME_PLAYBACK TaskTimer("Start playing on: %s", sys.deviceByIndex(_output_device).name() );

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
            sample_rate(),
            0,
            paNoFlag);//paClipOff);

    // Create (and (re)open) a new Stream:
    streamPlayback.reset( new portaudio::MemFunCallbackStream<Playback>(
            paramsPlayback,
            *this,
            &Signal::Playback::readBuffer) );

    _playback_itr = _data.first_buffer()->sample_offset;

    streamPlayback->start();

    } catch (const portaudio::PaException& x) {
        QMessageBox::warning( 0,
                     "Can't play sound",
                     x.what() );
        _data.reset();
    }
}

bool Playback::
        isStopped()
{
    return streamPlayback?!streamPlayback->isActive() || streamPlayback->isStopped():true;
}

bool Playback::
        isUnderfed()
{
    unsigned nAccumulated_samples = _data.number_of_samples();

    Signal::SamplesIntervalDescriptor expect = expected_samples();
    if (!_data.empty() && expect.isEmpty()) {
        TIME_PLAYBACK TaskTimer("Not underfed").suppressTiming();
        return false; // No more expected samples, not underfed
    }

	if (nAccumulated_samples < 0.1f*_data.sample_rate() || nAccumulated_samples < 3*_first_buffer_size ) {
        TIME_PLAYBACK TaskTimer("Underfed").suppressTiming();
        return true; // Haven't received much data, wait to do a better estimate
    }

    time_duration diff = _last_timestamp - _first_timestamp;
    float accumulation_time = diff.total_milliseconds() / (float)1000.f;

    // _first_timestamp is taken after the first buffer,
    // _last_timestamp is taken after the last buffer,
    // that means that accumulation_time is the time it took to accumulate all buffers except
    // the first buffer.
    float incoming_samples_per_sec = (nAccumulated_samples - _first_buffer_size) / accumulation_time;

    unsigned marker = _playback_itr;
    if (0==marker)
        marker = _data.first_buffer()->sample_offset;

    float time_left =
            (expect.intervals().back().last - marker) / (float)_data.sample_rate();

    Signal::SamplesIntervalDescriptor::Interval cov = expect.coveredInterval();
    float estimated_time_required =
            (cov.last - cov.first) / incoming_samples_per_sec;

    // Add small margin
    estimated_time_required *= 1.11f;

    // Return if the estimated time to receive all expected samples is greater than
    // the time it would take to play the remaining part of the data.
    // If it is, the sink is underfed.
    TIME_PLAYBACK TaskTimer("Time left %g %s %g estimated time required. %s underfed", time_left, time_left < estimated_time_required?"<":">=", estimated_time_required, time_left < estimated_time_required?"Is":"Not").suppressTiming();
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

    if (_playback_itr == _data.first_buffer()->sample_offset) {
        _startPlay_timestamp = microsec_clock::local_time();
    }

    pBuffer b = _data.readFixedLength( _playback_itr, framesPerBuffer );
    memcpy( buffer, b->waveform_data->getCpuMemory(), framesPerBuffer*sizeof(float) );
    _playback_itr += framesPerBuffer;

    if (_data.first_buffer()->sample_offset + _data.number_of_samples() + 10*2024/*framesPerBuffer*/ < _playback_itr ) {
        TIME_PLAYBACK TaskTimer tt("Reading %u, %u. Done at %u", _playback_itr, framesPerBuffer, _data.number_of_samples() );
        return paComplete;
    } else {
        if (_data.first_buffer()->sample_offset + _data.number_of_samples() < _playback_itr + framesPerBuffer) {
            TIME_PLAYBACK TaskTimer tt("Reading %u, %u. PAST END", _playback_itr, framesPerBuffer );
        } else {
            TIME_PLAYBACK TaskTimer tt("Reading %u, %u", _playback_itr, framesPerBuffer );
        }
    }

    return paContinue;
}

} // namespace Signal
