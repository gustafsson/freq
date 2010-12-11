#include "playback.h"

#include <iostream>
#include <stdexcept>
#include <QMutexLocker>
#include <QMessageBox>

#define TIME_PLAYBACK
//#define TIME_PLAYBACK if(0)

using namespace std;
using namespace boost::posix_time;

namespace Adapters {

Playback::
        Playback( int outputDevice )
:   _first_buffer_size(0),
    _output_device(0)
{
    portaudio::AutoSystem autoSys;
    portaudio::System &sys = portaudio::System::instance();

    static bool first = true;
    if (first) list_devices();

    if (0>outputDevice || outputDevice>sys.deviceCount()) {
        _output_device = sys.defaultOutputDevice().index();
    } else if ( sys.deviceByIndex(outputDevice).isInputOnlyDevice() ) {
        TaskTimer("Creating audio Playback. Requested audio device '%s' can only be used for input.",
                     sys.deviceByIndex(outputDevice).name()).suppressTiming();
        _output_device = sys.defaultOutputDevice().index();
    } else {
        _output_device = outputDevice;
    }

    if(first) TaskTimer("Creating audio Playback. Using device '%s' for audio output.",
                 sys.deviceByIndex(_output_device).name()).suppressTiming();

    reset();
    // first = false;
}


Playback::
        ~Playback()
{
    TaskTimer tt(__FUNCTION__);

    if (streamPlayback)
    {
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

    TaskTimer tt("Enumerating sound devices (count %d)", iNumDevices);
    for (portaudio::System::DeviceIterator i = sys.devicesBegin(); i != sys.devicesEnd(); ++i)
    {
        strDetails = "";
        if ((*i).isSystemDefaultInputDevice())
                strDetails += ", default input";
        if ((*i).isSystemDefaultOutputDevice())
                strDetails += ", default output";

        tt.info( "%d: %s, in=%d out=%d, %s%s",
                 (*i).index(), (*i).name(),
                 (*i).maxInputChannels(),
                 (*i).maxOutputChannels(),
                 (*i).hostApi().name(),
                 strDetails.c_str());

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
    if (_data.empty())
        return 0.f;

    // streamPlayback->time() doesn't seem to work (ubuntu 10.04)
    // float dt = streamPlayback?streamPlayback->time():0;
    
    time_duration d = microsec_clock::local_time() - _startPlay_timestamp;
    float dt = d.total_milliseconds()*0.001f;
    float t = dt;
    t += _data.first_buffer()->sample_offset / sample_rate();
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
        put( Signal::pBuffer buffer )
{
    TIME_PLAYBACK TaskTimer tt("Playback::put [%lu,%lu]", (unsigned long)buffer->sample_offset, (unsigned long)(buffer->sample_offset+buffer->number_of_samples()));

    _last_timestamp = microsec_clock::local_time();
    if (_data.empty())
	{
        _first_timestamp = _last_timestamp;
		_first_buffer_size = buffer->number_of_samples();
	}

    // Make sure the buffer is moved over to CPU memory.
    // (because the audio stream callback is executed from a different thread
    // it can't access the GPU memory)
    GpuCpuData<float>* bdata = buffer->waveform_data();
    bdata->getCpuMemory();
    bdata->freeUnused(); // relase GPU memory as well...

    _data.putExpectedSamples( buffer, _data.fetch_invalid_samples() );

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
    _max_found = 1;
    _min_found = -1;
}


bool Playback::
        isFinished()
{
    return false;
//    return _data.isFinished() && isStopped();
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
            &Playback::readBuffer) );

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
    return streamPlayback ? !streamPlayback->isActive() || streamPlayback->isStopped():true;
}


bool Playback::
        isUnderfed()
{
    unsigned nAccumulated_samples = _data.number_of_samples();

    if (!_data.empty() && _data.isFinished()) {
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

    long unsigned marker = _playback_itr;
    if (0==marker)
        marker = _data.first_buffer()->sample_offset;

    Signal::Interval cov = _data.fetch_invalid_samples().coveredInterval();
    float time_left =
            (cov.last - marker) / _data.sample_rate();

    float estimated_time_required = cov.count() / incoming_samples_per_sec;

    // Add small margin
    estimated_time_required *= 1.11f;

    // Return if the estimated time to receive all expected samples is greater than
    // the time it would take to play the remaining part of the data.
    // If it is, the sink is underfed.
    TIME_PLAYBACK TaskTimer("Time left %g %s %g estimated time required. %s underfed", time_left, time_left < estimated_time_required?"<":">=", estimated_time_required, time_left < estimated_time_required?"Is":"Not").suppressTiming();
    return time_left < estimated_time_required;
}


void Playback::
        restart_playback()
{
    if (streamPlayback)
    {
        TIME_PLAYBACK TaskTimer tt("Restaring playback");

        onFinished();
    }
}


void Playback::
        saturate( float* p, unsigned N )
{
    for (unsigned j=0; j<N; ++j)
    {
        if (p[j] > _max_found) _max_found = p[j];
        if (p[j] < _min_found) _min_found = p[j];

        p[j] = (p[j] - _min_found)/(_max_found - _min_found)*2-1;
    }
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

    Signal::pBuffer b = _data.readFixedLength( Signal::Interval(_playback_itr, _playback_itr+framesPerBuffer) );
    memcpy( buffer, b->waveform_data()->getCpuMemory(), framesPerBuffer*sizeof(float) );
    saturate( buffer, framesPerBuffer );
    _playback_itr += framesPerBuffer;

    if ((unsigned long)(_data.first_buffer()->sample_offset + _data.number_of_samples() + 10ul*2024/*framesPerBuffer*/) < _playback_itr ) {
        TIME_PLAYBACK TaskTimer("Playback::readBuffer %u, %u. Done at %u", _playback_itr, framesPerBuffer, _data.number_of_samples() ).suppressTiming();
        return paComplete;
    } else {
        if ( (unsigned long)(_data.first_buffer()->sample_offset + _data.number_of_samples()) < _playback_itr + framesPerBuffer) {
            TIME_PLAYBACK TaskTimer("Playback::readBuffer %u, %u. PAST END", _playback_itr, framesPerBuffer ).suppressTiming();
        } else {
            TIME_PLAYBACK TaskTimer("Playback::readBuffer Reading %u, %u", _playback_itr, framesPerBuffer ).suppressTiming();
        }
    }

    return paContinue;
}

} // namespace Adapters
