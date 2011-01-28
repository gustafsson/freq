#include "microphonerecorder.h"
#include "playback.h"

#include <iostream>
#include <memory.h>

#include <QMutexLocker>

#include <Statistics.h>

//#define TIME_MICROPHONERECORDER
#define TIME_MICROPHONERECORDER if(0)

using namespace std;

namespace Adapters {


MicrophoneRecorder::
        MicrophoneRecorder(int inputDevice)
            :
            input_device_(inputDevice),
            _sample_rate( 1 )
{
    init(); // fetch _sample_rate
    stopRecording();
}


void MicrophoneRecorder::
        init()
{
    _channel = 0;
    _offset = 0;

    static bool first = true;
    if (first) Playback::list_devices();

    TaskTimer tt("Creating MicrophoneRecorder for device %d", input_device_);
    portaudio::System &sys = portaudio::System::instance();

    bool has_input_device = false;
    for (int i=0; i < sys.deviceCount(); ++i)
    {
        if (!sys.deviceByIndex(i).isOutputOnlyDevice())
            has_input_device = true;
    }

    if (!has_input_device)
    {
        throw std::runtime_error("System didn't report any recording devices. Can't record.");
    }

    if (0>input_device_ || input_device_>sys.deviceCount()) {
        input_device_ = sys.defaultInputDevice().index();
    } else if ( sys.deviceByIndex(input_device_).isOutputOnlyDevice() ) {
        tt.getStream() << "Requested device '" << sys.deviceByIndex(input_device_).name() << "' can only be used for output";
        input_device_ = sys.defaultInputDevice().index();
    } else {
        ;
    }

    tt.getStream() << "Using device '" << sys.deviceByIndex(input_device_).name() << "' for audio input";

    portaudio::Device& device = sys.deviceByIndex(input_device_);
    _sample_rate = device.defaultSampleRate();

    unsigned channel_count = device.maxInputChannels();
    if (channel_count>2)
        channel_count = 2;
    tt.getStream() << "Opening recording input stream on '" << device.name() << "' with " << channel_count
                   << " channels, " << device.defaultSampleRate() << " samples/second"
                   << " and input latency " << device.defaultHighInputLatency() << " s";

    QMutexLocker lock(&_data_lock);
    _data.resize(channel_count);

    portaudio::DirectionSpecificStreamParameters inParamsRecord(
            device,
            channel_count, // channels
            portaudio::FLOAT32,
            false, // interleaved
//#ifdef __APPLE__ // TODO document why
            device.defaultHighInputLatency(),
//#else
//            device.defaultLowInputLatency(),
//#endif
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

    lock.unlock();
}


MicrophoneRecorder::~MicrophoneRecorder()
{
	TaskTimer tt("%s", __FUNCTION__);

    stopRecording();

    QMutexLocker lock(&_data_lock);
    for (unsigned i=0; i<_data.size(); ++i)
	{
		if (0<_data[i].length()) {
			TaskTimer tt("Releasing recorded data channel %u", i);
            _data[i].clear();
		}
	}
}

void MicrophoneRecorder::startRecording()
{
    TIME_MICROPHONERECORDER TaskInfo ti("MicrophoneRecorder::startRecording()");
    init();

    _stream_record->start();

    _start_recording = boost::posix_time::microsec_clock::local_time();
    _offset = length();
}

void MicrophoneRecorder::stopRecording()
{
    TIME_MICROPHONERECORDER TaskInfo ti("MicrophoneRecorder::stopRecording()");
    if (_stream_record) {
        //stop could hang the ui (codaset #24)
        //_stream_record->isStopped()? void(): _stream_record->stop();
        _stream_record->isStopped()? void(): _stream_record->abort();
        _stream_record->close();
        _stream_record.reset();
    }
}

bool MicrophoneRecorder::isStopped()
{
    return _stream_record?_stream_record->isStopped():true;
}

Signal::pBuffer MicrophoneRecorder::
        read( const Signal::Interval& I )
{
    QMutexLocker lock(&_data_lock);
    // TODO why? return _data[channel].readFixedLength( I );
    return _data[_channel].read( I );
}

float MicrophoneRecorder::
        sample_rate()
{
    return _sample_rate;
}

long unsigned MicrophoneRecorder::
        number_of_samples()
{
    QMutexLocker lock(&_data_lock);
    long unsigned N = _data[_channel].number_of_samples();
    return N;
}

unsigned MicrophoneRecorder::
        num_channels()
{
    QMutexLocker lock(&_data_lock);
#ifdef SAWE_MONO
    return _data.size()?1:0;
#else
    return _data.size();
#endif
}

void MicrophoneRecorder::
        set_channel(unsigned channel)
{
    //TIME_MICROPHONERECORDER TaskTimer("MicrophoneRecorder::set_channel(%u)", channel).suppressTiming();
    BOOST_ASSERT( channel < num_channels() );
    this->_channel = channel;
}


unsigned MicrophoneRecorder::
        get_channel()
{
    return _channel;
}


float MicrophoneRecorder::
        time()
{
    boost::posix_time::time_duration d = boost::posix_time::microsec_clock::local_time() - _start_recording;
    float dt = d.total_milliseconds()*0.001f;
    return dt + _offset;
}


int MicrophoneRecorder::
        writeBuffer(const void *inputBuffer,
                 void * /*outputBuffer*/,
                 unsigned long framesPerBuffer,
                 const PaStreamCallbackTimeInfo * /*timeInfo*/,
                 PaStreamCallbackFlags /*statusFlags*/)
{
    TIME_MICROPHONERECORDER TaskTimer tt("MicrophoneRecorder::writeBuffer(%u new samples) inputBuffer = %p", framesPerBuffer, inputBuffer);
    const float **in = (const float **)inputBuffer;

	long unsigned offset = number_of_samples();
    QMutexLocker lock(&_data_lock);

    for (unsigned i=0; i<_data.size(); ++i)
    {
        const float *buffer = in[i];
        Signal::pBuffer b( new Signal::Buffer(0, framesPerBuffer, sample_rate() ) );
        memcpy ( b->waveform_data()->getCpuMemory(),
                 buffer,
                 framesPerBuffer*sizeof(float) );

        b->sample_offset = offset;
        b->sample_rate = sample_rate();

        TIME_MICROPHONERECORDER TaskInfo ti("Interval: %s, [%g, %g) s",
                                            b->getInterval().toString().c_str(),
                                            b->getInterval().first / b->sample_rate,
                                            b->getInterval().last / b->sample_rate );

        _data[i].put( b );
    }

    lock.unlock();

    _postsink.invalidate_samples( Signal::Interval( offset, offset + framesPerBuffer ));

    return paContinue;
}

} // namespace Adapters
