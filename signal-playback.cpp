#include "signal-playback.h"
#include <iostream>
#include <stdexcept>
#include <boost/foreach.hpp>

using namespace std;

namespace Signal {

Playback::Playback( /*int outputDevice*/ )
:   _playback_itr(0)
{
    portaudio::AutoSystem autoSys;
    portaudio::System &sys = portaudio::System::instance();

    cout << "Using system default input/output devices..." << endl;
    unsigned iInputDevice	= sys.defaultInputDevice().index();
    unsigned iOutputDevice	= sys.defaultOutputDevice().index();
    cout << iInputDevice << " " << iOutputDevice << endl;
}

Playback::~Playback()
{
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

    std::cout << "Number of devices = " << iNumDevices << std::endl;

    cout << "Using system default input/output devices..." << endl;
    unsigned iInputDevice	= sys.defaultInputDevice().index();
    unsigned iOutputDevice	= sys.defaultOutputDevice().index();
    cout << " input device " << iInputDevice << endl
         << " output device " << iOutputDevice << endl;

    //		portaudio::Device inDevice = portaudio::Device(sys.defaultInputDevice());

    //		portaudio::Device& inDevice 	= sys.deviceByIndex(iInputDevice);
    //portaudio::Device& outDevice 	= sys.deviceByIndex(iOutputDevice);

    for (portaudio::System::DeviceIterator i = sys.devicesBegin(); i != sys.devicesEnd(); ++i)
    {
        strDetails = "";
        if ((*i).isSystemDefaultInputDevice())
                strDetails += ", default input";
        if ((*i).isSystemDefaultOutputDevice())
                strDetails += ", default output";

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

void Playback::put( pBuffer buffer )
{
    if (!_cache.empty()) if (_cache[0].buffer->sample_rate != buffer->sample_rate) {
        throw std::logic_error(std::string(__FUNCTION__) + " sample rate is different from previous sample rate" );
    }

    BufferSlot slot = { buffer, clock() };

    if (streamPlayback) {
        // not thread-safe, could stop playing if _cache is empty after isPlaying returned true and before _cache.push_back returns
        _cache.push_back( slot );
        if (streamPlayback->isStopped()) streamPlayback->start();
        return;
    }

    // estimate speed of input, in samples per second
    unsigned nAccumulated_samples = nAccumulatedSamples();

    _cache.push_back( slot );

    clock_t first = _cache.front().timestamp;
    clock_t last = _cache.back().timestamp;

    float accumulation_time = (last-first) / (float)CLOCKS_PER_SEC;
    float incoming_samples_per_sec = nAccumulated_samples / accumulation_time;
    float time_left = (nAccumulated_samples - _playback_itr + expected_samples_left()) / (float)buffer->sample_rate;
    float estimated_time_required = expected_samples_left() / incoming_samples_per_sec;

    unsigned x = expected_samples_left();
    if (x < buffer->number_of_samples() )
        x = 0;
    else
        x -= buffer->number_of_samples();
    expected_samples_left( x );

    if (x == 0 || time_left < estimated_time_required )
    {
        // start playing
        portaudio::System &sys = portaudio::System::instance();
        unsigned iOutputDevice	= sys.defaultOutputDevice().index();

        // Set up the parameters required to open a (Callback)Stream:
        portaudio::DirectionSpecificStreamParameters outParamsPlayback(
                sys.deviceByIndex(iOutputDevice),
                1, // mono sound
                portaudio::FLOAT32,
                false,
                sys.deviceByIndex(iOutputDevice).defaultLowOutputLatency(),
                NULL);
        portaudio::StreamParameters paramsPlayback(
                portaudio::DirectionSpecificStreamParameters::null(),
                outParamsPlayback,
                buffer->sample_rate,
                buffer->number_of_samples(),
                paClipOff);

        // Create (and open) a new Stream, using the SineGenerator::generate function as a callback:
        cout << "Opening beep output stream on: " << sys.deviceByIndex(iOutputDevice).name() << endl;
        streamPlayback.reset( new portaudio::MemFunCallbackStream<Playback>(
                paramsPlayback,
                *this,
                &Signal::Playback::readBuffer) );

        streamPlayback->start();
    }
}

void Playback::
    reset()
{
    _cache.clear();
    _playback_itr = 0;
    streamPlayback.reset();
}

unsigned Playback::
    nAccumulatedSamples()
{
    // count previous samples
    unsigned nAccumulated_samples = 0;
    BOOST_FOREACH( const BufferSlot& s, _cache ) {
        nAccumulated_samples += s.buffer->number_of_samples();
    }
    return nAccumulated_samples;
}

int Playback::
        readBuffer(const void */*inputBuffer*/,
                 void *outputBuffer,
                 unsigned long framesPerBuffer,
                 const PaStreamCallbackTimeInfo */*timeInfo*/,
                 PaStreamCallbackFlags /*statusFlags*/)
{
    BOOST_ASSERT( outputBuffer );
    float **out = static_cast<float **>(outputBuffer);
    float *buffer = out[0];

    PaStreamCallbackResult r = paComplete;
    unsigned iBuffer = 0;
    unsigned nAccumulated_samples = 0;

    while ( 0 < framesPerBuffer )
    {
        // count previous samples
        for (; iBuffer<_cache.size(); iBuffer++) {
            const BufferSlot& s = _cache[iBuffer];
            if (_playback_itr < nAccumulated_samples + s.buffer->number_of_samples() )
                break;
            nAccumulated_samples += s.buffer->number_of_samples();
        }

        if (iBuffer >= _cache.size())
        {
            memset(buffer, 0, framesPerBuffer*sizeof(float));
            framesPerBuffer = 0;
        } else {
            r = paContinue;

            const BufferSlot& s = _cache[iBuffer];

            unsigned nSamples_to_copy = nAccumulated_samples + s.buffer->number_of_samples() - _playback_itr;
            if (framesPerBuffer < nSamples_to_copy )
                nSamples_to_copy = framesPerBuffer;

            memcpy( buffer,
                    &s.buffer->waveform_data->getCpuMemory()[ _playback_itr - nAccumulated_samples ],
                    nSamples_to_copy*sizeof(float));

            buffer += nSamples_to_copy;
            framesPerBuffer -= nSamples_to_copy;
            _playback_itr += nSamples_to_copy;
        }
    }
    return r;
}

} // namespace Signal
