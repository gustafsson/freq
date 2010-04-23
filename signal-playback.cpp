#include "signal-playback.h"
#include <iostream>
#include <stdexcept>
#include <boost/foreach.hpp>

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
    if (!_cache.empty()) if (_cache[0].buffer->sample_rate != buffer->sample_rate) {
        throw std::logic_error(std::string(__FUNCTION__) + " sample rate is different from previous sample rate" );
    }

    _first_invalid_sample = buffer->sample_offset + buffer->number_of_samples();

    BufferSlot slot = { buffer, clock() };

    if (streamPlayback) {
        // not thread-safe, could stop playing if _cache is empty after isPlaying returned true and before _cache.push_back returns
        _cache.push_back( slot );

        if (streamPlayback->isStopped()) {
            // start over
            streamPlayback->start();
        }
        return;
    }

    _cache.push_back( slot );

    unsigned x = expected_samples_left();
    if (x < buffer->number_of_samples() )
        x = 0;
    else
        x -= buffer->number_of_samples();
    expected_samples_left( x );

    if (isUnderfed() ) {
        //  Wait for more data

    } else {
        // start playing
        portaudio::System &sys = portaudio::System::instance();

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

        // Create (and open) a new Stream, using the SineGenerator::generate function as a callback:
        //cout << "Opening beep output stream on: " << sys.deviceByIndex(iOutputDevice).name() << endl;
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
    if (streamPlayback) if (!streamPlayback->isStopped()) streamPlayback->stop();
    _cache.clear();
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
    // count previous samples
    unsigned nAccumulated_samples = 0;
    BOOST_FOREACH( const BufferSlot& s, _cache ) {
        nAccumulated_samples += s.buffer->number_of_samples();
    }
    return nAccumulated_samples;
}

pBuffer Playback::
        first_buffer()
{
    if (_cache.empty())
        return pBuffer();
    return _cache[0].buffer;
}

bool Playback::
        isStopped()
{
    return streamPlayback?!streamPlayback->isActive() || streamPlayback->isStopped():true;
}

bool Playback::
        isUnderfed()
{
    if (1>=_cache.size())
        return true;

    unsigned nAccumulated_samples = nAccumulatedSamples();

    clock_t first = _cache.front().timestamp;
    clock_t last = _cache.back().timestamp;


    float accumulation_time = (last-first) / (float)CLOCKS_PER_SEC;
    float incoming_samples_per_sec = (nAccumulated_samples - _cache.front().buffer->number_of_samples()) / accumulation_time;
    float time_left = (nAccumulated_samples - _playback_itr + expected_samples_left()) / (float)_cache[0].buffer->sample_rate;
    float estimated_time_required = expected_samples_left() / incoming_samples_per_sec;

    if (expected_samples_left() == 0 || time_left < estimated_time_required )
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

            float *source = s.buffer->waveform_data->getCpuMemory();
            memcpy( buffer,
                    &source[ _playback_itr - nAccumulated_samples ],
                    nSamples_to_copy*sizeof(float));

            buffer += nSamples_to_copy;
            framesPerBuffer -= nSamples_to_copy;
            _playback_itr += nSamples_to_copy;
        }
    }
    return r;
}

} // namespace Signal
