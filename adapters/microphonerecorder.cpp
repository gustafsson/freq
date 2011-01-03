#include "microphonerecorder.h"
#include "playback.h"

#include <iostream>
#include <memory.h>

#include <QMutexLocker>

#include <Statistics.h>

#define TIME_MICROPHONERECORDER
//#define TIME_MICROPHONERECORDER if(0)

using namespace std;

namespace Adapters {

/**
  TODO document why this class is needed.
  */
class OperationProxy: public Signal::Operation
{
public:
    OperationProxy(Signal::Operation* p)
        :
        Operation(Signal::pOperation()),
        p(p)
    {}

    virtual Signal::pBuffer read( const Signal::Interval& I ) { return p->read( I ); }
    virtual float sample_rate() { return p->sample_rate(); }

private:
    Signal::Operation*p;
};

MicrophoneRecorder::MicrophoneRecorder(int inputDevice)
    :
    channel(0)
{
    static bool first = true;
    if (first) Playback::list_devices();

    TaskTimer tt("Creating MicrophoneRecorder for device %d", inputDevice);
    portaudio::System &sys = portaudio::System::instance();

    if (0>inputDevice || inputDevice>sys.deviceCount()) {
        inputDevice = sys.defaultInputDevice().index();
    } else if ( sys.deviceByIndex(inputDevice).isOutputOnlyDevice() ) {
        tt.getStream() << "Requested device '" << sys.deviceByIndex(inputDevice).name() << "' can only be used for output";
        inputDevice = sys.defaultInputDevice().index();
    } else {
        inputDevice = inputDevice;
    }

    tt.getStream() << "Using device '" << sys.deviceByIndex(inputDevice).name() << "' for audio input";

    portaudio::Device& device = sys.deviceByIndex(inputDevice);

    unsigned channel_count = device.maxInputChannels();
    if (channel_count>2)
        channel_count = 2;
    tt.getStream() << "Opening recording input stream on '" << device.name() << "' with " << channel_count << " channels";

    QMutexLocker lock(&_data_lock);
    _data.resize(channel_count);

    portaudio::DirectionSpecificStreamParameters inParamsRecord(
            device,
            channel_count, // channels
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

    Signal::pOperation proxy(new OperationProxy(&_data[0]));
    _postsink.source( proxy );
}

MicrophoneRecorder::~MicrophoneRecorder()
{
	TaskTimer tt("%s", __FUNCTION__);
    if (_stream_record) {
        _stream_record->isStopped()? void(): _stream_record->stop();
        _stream_record->isStopped()? void(): _stream_record->abort();
        _stream_record->close();
    }

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

Signal::pBuffer MicrophoneRecorder::
        read( const Signal::Interval& I )
{
    QMutexLocker lock(&_data_lock);
    return _data[channel].readFixedLength( I );
}

float MicrophoneRecorder::
        sample_rate()
{
    return _stream_record->sampleRate();
}

long unsigned MicrophoneRecorder::
        number_of_samples()
{
    QMutexLocker lock(&_data_lock);
    return _data[channel].number_of_samples();
}

unsigned MicrophoneRecorder::
        num_channels()
{
    QMutexLocker lock(&_data_lock);
    return _data.size()?1:0;
}

void MicrophoneRecorder::
        set_channel(unsigned channel)
{
    //TIME_MICROPHONERECORDER TaskTimer("MicrophoneRecorder::set_channel(%u)", channel).suppressTiming();
    BOOST_ASSERT( channel < num_channels() );
    this->channel = channel;
}


unsigned MicrophoneRecorder::
        get_channel()
{
    return channel;
}


int MicrophoneRecorder::
        writeBuffer(const void *inputBuffer,
                 void * /*outputBuffer*/,
                 unsigned long framesPerBuffer,
                 const PaStreamCallbackTimeInfo * /*timeInfo*/,
                 PaStreamCallbackFlags /*statusFlags*/)
{
    TIME_MICROPHONERECORDER TaskTimer tt("MicrophoneRecorder::writeBuffer(%u new samples)", framesPerBuffer);
    BOOST_ASSERT( inputBuffer );
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

        TIME_MICROPHONERECORDER TaskInfo ti("Interval: %s", b->getInterval().toString().c_str());
        //Statistics<float>(b->waveform_data());

        _data[i].put( b );
    }

    lock.unlock();

	_postsink.invalidate_samples( Signal::Interval( offset, offset + framesPerBuffer ));

    return paContinue;
}

} // namespace Adapters
