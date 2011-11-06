#include "playback.h"

#include <iostream>
#include <stdexcept>
#include <QMessageBox>

#include "cpumemorystorage.h"

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
    _data.setNumChannels(0);

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

    if(first)
    {
        TaskInfo tt("Creating audio Playback. Using device '%s' (%d) for audio output.",
                 sys.deviceByIndex(_output_device).name(), _output_device);
        if (_output_device != outputDevice)
            tt.tt().getStream() << " Requested device was number " << outputDevice;
    }

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


/*static*/ std::list<Playback::DeviceInfo> Playback::
        get_devices()
{
    // initialize enumeration of devices if it hasn't been done already
    portaudio::AutoSystem autoSys;

    portaudio::System &sys = portaudio::System::instance();

    std::list<DeviceInfo> devices;

    for (portaudio::System::DeviceIterator i = sys.devicesBegin(); i != sys.devicesEnd(); ++i)
    {
        DeviceInfo d;
        d.name = (*i).name();
        d.name2 = (*i).hostApi().name();
        d.inputChannels = (*i).maxInputChannels();
        d.outputChannels = (*i).maxOutputChannels();
        d.isDefaultIn = (*i).isSystemDefaultInputDevice();
        d.isDefaultOut = (*i).isSystemDefaultOutputDevice();
        d.index = (*i).index();

        std::string strDetails = (*i).hostApi().name();
        if ((*i).isSystemDefaultInputDevice())
               strDetails += ", default input";
        if ((*i).isSystemDefaultOutputDevice())
               strDetails += ", default output";
        d.name2 = strDetails;

        devices.push_back( d );
    }

    return devices;
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

    // if !isPaused() the stream has started, but it hasn't read anything yet if this holds, and _startPlay_timestamp is not set
    if (isPaused() || _playback_itr == _data.getInterval().first )
    {
        return _playback_itr / sample_rate();
    }

    // streamPlayback->time() doesn't seem to work (ubuntu 10.04)
    // float dt = streamPlayback?streamPlayback->time():0;
    
    time_duration d = microsec_clock::local_time() - _startPlay_timestamp;
    float dt = d.total_milliseconds()*0.001f;
    float t = dt;
    t += _data.getInterval().first / sample_rate();
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
    TIME_PLAYBACK TaskTimer tt("Playback::put %s", buffer->getInterval().toString().c_str());
    _last_timestamp = microsec_clock::local_time();

    if (_data.empty())
    {
        const Signal::Interval I = buffer->getInterval();
        _first_timestamp = _last_timestamp;
        _first_buffer_size = I.count();

        // Discard zeros in the beginning of the signal
/*        float* p = buffer->waveform_data()->getCpuMemory();
        Signal::IntervalType i;
        for (i=0; i<I.count(); ++i)
        {
            if (fabsf(p[i]) > 0.001)
                break;
        }

        if (i>0)
        {
            Signal::Intervals is = _data.invalid_samples();
            _data.clear();
            _data.invalidate_samples( is - Signal::Interval(I.first, I.first + i) );

            Signal::Interval rI( I.first + i, I.last );

            if (0==rI.count())
                return;

            buffer = Signal::BufferSource( buffer ).readFixedLength( rI );
        }*/
	}

    // Make sure the buffer is moved over to CPU memory and that GPU memory is released
    // (because the audio stream callback is executed from a different thread
    // it can't access the GPU memory)
    buffer->waveform_data()->OnlyKeepOneStorage<CpuMemoryStorage>();

    _data.putExpectedSamples( buffer );

    if (streamPlayback)
    {
        if (streamPlayback->isActive()) {
            TIME_PLAYBACK TaskInfo("Is playing");
        } else {
            TIME_PLAYBACK TaskInfo("Is paused");
        }
        return;
    }

    onFinished();
}


void Playback::
        stop()
{
    if (streamPlayback)
    {
        // streamPlayback->stop will invoke a join with readBuffer
        if (!streamPlayback->isStopped())
            streamPlayback->stop();
    }

    _playback_itr = _data.getInterval().last;
    _max_found = 1;
    _min_found = -1;
}


void Playback::
        reset()
{
    stop();

    if (streamPlayback)
        streamPlayback.reset();

    _playback_itr = 0;

    _data.clear();
}


bool Playback::
        deleteMe()
{
    // Keep cache
    return false;

    // Discard cache
    //return _data.deleteMe() && isStopped();
}


void Playback::
        invalidate_samples( const Signal::Intervals& s )
{
    // If the CwtFilter runs out of memory and changes the number of scales per
    // octave it will invalidate all samples. Discard that and keep the samples
    // we've received for playback so far.
    bool invalidates_all = (s == getInterval());
    bool has_been_initialized = _data.invalid_samples();

    if (has_been_initialized && invalidates_all)
        return;

    // Don't bother recomputing stuff we've already played
    Signal::Interval whatsLeft(
                        _playback_itr,
                        Signal::Interval::IntervalType_MAX);

    _data.invalidate_samples( s & whatsLeft );

    if (0 == _data.num_channels())
        _data.setNumChannels( source()->num_channels() );
}


unsigned Playback::
        num_channels()
{
    return _data.num_channels();
}


void Playback::
        set_channel(unsigned c)
{
    // If more channels are requested for playback than the output device has channels available,
    // then let the last channel requested go into the last channel available and discard other
    // superfluous channels
    if (c >= num_channels())
        c = num_channels() - 1;

    _data.set_channel( c );
}


Signal::Intervals Playback::
        invalid_samples()
{
    return _data.invalid_samples();
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

    unsigned requested_number_of_channels = num_channels();
    unsigned available_channels = sys.deviceByIndex(_output_device).maxOutputChannels();

    if (available_channels<requested_number_of_channels)
    {
        requested_number_of_channels = available_channels;
        _data.setNumChannels( requested_number_of_channels );
    }

    // Set up the parameters required to open a (Callback)Stream:
    portaudio::DirectionSpecificStreamParameters outParamsPlayback(
            sys.deviceByIndex(_output_device),
            requested_number_of_channels,
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

    _playback_itr = _data.getInterval().first;

    streamPlayback->start();

    } catch (const portaudio::PaException& x) {
        QMessageBox::warning( 0,
                     "Can't play sound",
                     x.what() );
        _data.clear();
    }
}


bool Playback::
        isStopped()
{
    //return streamPlayback ? !streamPlayback->isActive() || streamPlayback->isStopped():true;
    bool isActive = streamPlayback ? streamPlayback->isActive() : false;
    //bool isStopped = streamPlayback ? streamPlayback->isStopped() : true;
    // isActive and isStopped might both be false at the same time
    bool paused = isPaused();
    return streamPlayback ? !isActive && !paused : true;
}


bool Playback::
        isPaused()
{
    bool isStopped = streamPlayback ? streamPlayback->isStopped() : true;

    bool read_past_end = _data.empty() ? true : _playback_itr >= _data.getInterval().last;

    return isStopped && !read_past_end;
}


bool Playback::
        hasReachedEnd()
{
    if (_data.empty())
        return false;

    if (_data.invalid_samples())
        return false;

    return time()*_data.sample_rate() > _data.getInterval().last;
}


bool Playback::
        isUnderfed()
{
    unsigned nAccumulated_samples = _data.number_of_samples();

    if (!_data.empty() && !_data.invalid_samples()) {
        TIME_PLAYBACK TaskInfo("Not underfed");
        return false; // No more expected samples, not underfed
    }

    if (nAccumulated_samples < 0.1f*_data.sample_rate() || nAccumulated_samples < 3*_first_buffer_size ) {
        TIME_PLAYBACK TaskInfo("Underfed");
        return true; // Haven't received much data, wait to do a better estimate
    }

    time_duration diff = _last_timestamp - _first_timestamp;
    float accumulation_time = diff.total_milliseconds() * 0.001f;

    // _first_timestamp is taken after the first buffer,
    // _last_timestamp is taken after the last buffer,
    // that means that accumulation_time is the time it took to accumulate all buffers except
    // the first buffer.
    float incoming_samples_per_sec = (nAccumulated_samples - _first_buffer_size) / accumulation_time;

    long unsigned marker = _playback_itr;
    if (0==marker)
        marker = _data.getInterval().first;

    Signal::Interval cov = _data.invalid_samples().coveredInterval();
    float time_left =
            (cov.last - marker) / _data.sample_rate();

    float estimated_time_required = cov.count() / incoming_samples_per_sec;

    // Add small margin
    estimated_time_required *= 1.11f;

    // Return if the estimated time to receive all expected samples is greater than
    // the time it would take to play the remaining part of the data.
    // If it is, the sink is underfed.
    TIME_PLAYBACK TaskInfo("Time left %g %s %g estimated time required. %s underfed", time_left, time_left < estimated_time_required?"<":">=", estimated_time_required, time_left < estimated_time_required?"Is":"Not");
    bool underfed = false;
    underfed |= time_left < estimated_time_required;

    // Also, check that we keep a margin of 3 buffers
    underfed |= marker + 3*_first_buffer_size > cov.first;

    return underfed;
}


void Playback::
        pausePlayback(bool pause)
{
    if (!streamPlayback)
        return;

    if (pause)
    {
        _playback_itr = time()*sample_rate();

        if (streamPlayback->isActive() && !streamPlayback->isStopped())
            streamPlayback->stop();
    }
    else
    {
        if (!isPaused() || _data.empty())
            return;

        _startPlay_timestamp = microsec_clock::local_time();
        _startPlay_timestamp -= time_duration(0, 0, 0, (_playback_itr-_data.getInterval().first)/sample_rate()*time_duration::ticks_per_second() );

        streamPlayback->start();
    }
}


void Playback::
        restart_playback()
{
    if (streamPlayback)
    {
        TIME_PLAYBACK TaskTimer tt("Restaring playback");

        _playback_itr = 0;
        _max_found = 1;
        _min_found = -1;

        onFinished();
    }
}


void Playback::
        normalize( float* p, unsigned N )
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
    if (!_data.empty() && _playback_itr == _data.getInterval().first) {
        _startPlay_timestamp = microsec_clock::local_time();
    }

    float **out = static_cast<float **>(outputBuffer);
    for (unsigned c=0; c<num_channels(); ++c)
    {
        float *buffer = out[c];

        Signal::pBuffer b = _data.channel(c).readFixedLength( Signal::Interval(_playback_itr, _playback_itr+framesPerBuffer) );
        ::memcpy( buffer, b->waveform_data()->getCpuMemory(), framesPerBuffer*sizeof(float) );
        normalize( buffer, framesPerBuffer );
    }

    _playback_itr += framesPerBuffer;

    const char* msg = "";
    int ret = paContinue;
    if ((unsigned long)(_data.getInterval().last + framesPerBuffer) < _playback_itr ) {
        msg = ". DONE";
        ret = paComplete;
    } else {
        if ( (unsigned long)(_data.getInterval().last) < _playback_itr ) {
            msg = ". PAST END";
            // TODO if !_data.invalid_samples().empty() should pause playback here and continue when data is made available
        } else {
        }
    }

    float FS;
    TIME_PLAYBACK FS = _data.sample_rate();
    TIME_PLAYBACK TaskInfo("Playback::readBuffer Reading [%u, %u)%u# from %u. [%g, %g)%g s%s",
                           _playback_itr, _playback_itr+framesPerBuffer, framesPerBuffer,
                           _data.number_of_samples(),
                           _playback_itr/ FS, (_playback_itr + framesPerBuffer)/ FS,
                           framesPerBuffer/ FS, msg);

    return ret;
}

} // namespace Adapters
