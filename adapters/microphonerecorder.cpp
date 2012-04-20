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
            _is_interleaved(false)
{
    std::stringstream ss;
    ::srand(::time(0));
    unsigned int s = rand()^(rand() << 8) ^ (rand() << 16) ^ (rand() << 24);

    ss << "recording_" << s << ".wav";

    init(); // fetch _sample_rate
    stopRecording(); // delete _stream_record
}


void MicrophoneRecorder::
        init()
{
    try
    {
        _offset = 0;
        _sample_rate = 1;

        static bool first = true;
        if (first) Playback::list_devices();

        TaskTimer tt("Creating MicrophoneRecorder for device %d", input_device_);
        portaudio::System &sys = portaudio::System::instance();

        _has_input_device = false;
        for (int i=0; i < sys.deviceCount(); ++i)
        {
            if (!sys.deviceByIndex(i).isOutputOnlyDevice())
                _has_input_device = true;
        }

        if (!_has_input_device)
        {
            TaskInfo("System didn't report any recording devices. Can't record.");
            return;
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
        _data.setNumChannels(channel_count);
        _rolling_mean.resize(channel_count);
        for (unsigned i=0; i<channel_count; ++i)
            _rolling_mean[i] = 0;

        for (int interleaved=0; interleaved<2; ++interleaved)
        {
            _is_interleaved = interleaved!=0;

            portaudio::DirectionSpecificStreamParameters inParamsRecord(
                    device,
                    channel_count, // channels
                    portaudio::FLOAT32,
                    interleaved, // interleaved
        //#ifdef __APPLE__ // TODO document why
                    device.defaultHighInputLatency(),
        //#else
        //            device.defaultLowInputLatency(),
        //#endif
                    NULL);

            PaError err = Pa_IsFormatSupported(inParamsRecord.paStreamParameters(), 0, device.defaultSampleRate());
            bool fmtok = err==paFormatIsSupported;
            if (!fmtok)
                continue;

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
            break;
        }

        if (!_stream_record)
            _has_input_device = false;
    }
    catch (const portaudio::PaException& x)
    {
        TaskInfo("a2 MicrophoneRecorder init error: %s %s (%d)\nMessage: %s",
                 vartype(x).c_str(), x.paErrorText(), x.paError(), x.what());
        _has_input_device = false;
    }
    catch (const portaudio::PaCppException& x)
    {
        TaskInfo("b2 MicrophoneRecorder init error: %s (%d)\nMessage: %s",
                 vartype(x).c_str(), x.specifier(), x.what());
        _has_input_device = false;
    }
}


MicrophoneRecorder::~MicrophoneRecorder()
{
	TaskTimer tt("%s", __FUNCTION__);

    stopRecording();

    QMutexLocker lock(&_data_lock);
    if (0<_data.length()) {
        TaskTimer tt("Releasing %s recorded data in %u channels",
                     _data.lengthLongFormat().c_str(),
                     _data.num_channels());
        _data.clear();
    }
}

void MicrophoneRecorder::startRecording()
{
    TIME_MICROPHONERECORDER TaskInfo ti("MicrophoneRecorder::startRecording()");
    init();

    if (!canRecord())
    {
        TaskInfo("MicrophoneRecorder::startRecording() cant start recording");
        return;
    }

    // call length() and update _offset before starting recording because
    // length() uses {number_of_samples - time() - _offset} while recording.
    _offset = length();

    try
    {
        _stream_record->start();
    }
    catch (const portaudio::PaException& x)
    {
        TaskInfo("startRecording error: %s %s (%d)\nMessage: %s",
                 vartype(x).c_str(), x.paErrorText(), x.paError(), x.what());
        _has_input_device = false;
        return;
    }
    catch (const portaudio::PaCppException& x)
    {
        TaskInfo("startRecording error: %s (%d)\nMessage: %s",
                 vartype(x).c_str(), x.specifier(), x.what());
        _has_input_device = false;
        return;
    }

    _start_recording = boost::posix_time::microsec_clock::local_time();
}

void MicrophoneRecorder::stopRecording()
{
    TIME_MICROPHONERECORDER TaskInfo ti("MicrophoneRecorder::stopRecording()");
    if (_stream_record) {
        try
        {
        TaskInfo ti("Trying to stop recording on %s", deviceName().c_str());
        //stop could hang the ui (codaset #24)
        //_stream_record->isStopped()? void(): _stream_record->stop();
        _stream_record->isStopped()? void(): _stream_record->abort();
        _stream_record->close();
        _stream_record.reset();
        }
        catch (const portaudio::PaException& x)
        {
            TaskInfo("stopRecording error: %s %s (%d)\nMessage: %s",
                     vartype(x).c_str(), x.paErrorText(), x.paError(), x.what());
            _has_input_device = false;
        }
        catch (const portaudio::PaCppException& x)
        {
            TaskInfo("stopRecording error: %s (%d)\nMessage: %s",
                     vartype(x).c_str(), x.specifier(), x.what());
            _has_input_device = false;
        }
    }
}

bool MicrophoneRecorder::isStopped()
{
    return _stream_record?_stream_record->isStopped():true;
}

std::string MicrophoneRecorder::
        deviceName()
{
    portaudio::System &sys = portaudio::System::instance();
    int d = input_device_;
    if (d<0 || d>=sys.deviceCount())
        d = sys.defaultInputDevice().index();

    return sys.deviceByIndex(d).name();
}

void MicrophoneRecorder::
        setProjectName(std::string project, int i)
{
    stringstream ss;
    string device = deviceName();
    replace(device.begin(), device.end(), ' ', '_');
    ss << project << "_" << i << "_" << device << ".wav";
    _filename = ss.str();
}


bool MicrophoneRecorder::
        canRecord()
{
    return _has_input_device;
}


void MicrophoneRecorder::
        changeInputDevice( int inputDevice )
{
    bool isRecording = !isStopped();
    stopRecording();

    input_device_ = inputDevice;
    init();

    if (isRecording)
        startRecording();
}


std::string MicrophoneRecorder::
        name()
{
    return "Recording mic " + deviceName();
}


float MicrophoneRecorder::
        sample_rate()
{
    return _sample_rate;
}


int MicrophoneRecorder::
        writeBuffer(const void *inputBuffer,
                 void * /*outputBuffer*/,
                 unsigned long framesPerBuffer,
                 const PaStreamCallbackTimeInfo * /*timeInfo*/,
                 PaStreamCallbackFlags /*statusFlags*/)
{
    TIME_MICROPHONERECORDER TaskTimer tt("MicrophoneRecorder::writeBuffer(%u new samples) inputBuffer = %p", framesPerBuffer, inputBuffer);

    Signal::IntervalType offset = actual_number_of_samples();
    QMutexLocker lock(&_data_lock);
    _last_update = boost::posix_time::microsec_clock::local_time();
    unsigned prev_channel = _data.get_channel();
    unsigned num_channels = _data.num_channels();

    for (unsigned i=0; i<num_channels; ++i)
    {
        Signal::pBuffer b( new Signal::Buffer(0, framesPerBuffer, sample_rate() ) );
        float* p = b->waveform_data()->getCpuMemory();

        if (_is_interleaved)
        {
            const float *in = (const float *)inputBuffer;
            for (unsigned j=0; j<framesPerBuffer; ++j)
                p[j] = in[j*num_channels + i];
        }
        else
        {
            const float **in = (const float **)inputBuffer;
            const float *buffer = in[i];
            for (unsigned j=0; j<framesPerBuffer; ++j)
                p[j] = buffer[j];
        }

        float& mean = _rolling_mean[i];
        for (unsigned j=0; j<framesPerBuffer; ++j)
        {
            float v = p[j];
            p[j] = v - mean;
            mean = mean*0.99999f + v*0.00001f;
        }

//        memcpy ( b->waveform_data()->getCpuMemory(),
//                 buffer,
//                 framesPerBuffer*sizeof(float) );

        b->sample_offset = offset;
        b->sample_rate = sample_rate();

        TIME_MICROPHONERECORDER TaskInfo ti("Interval: %s, [%g, %g) s",
                                            b->getInterval().toString().c_str(),
                                            b->getInterval().first / b->sample_rate,
                                            b->getInterval().last / b->sample_rate );

        _data.set_channel(i);
        _data.put( b );
    }

    _data.set_channel(prev_channel);
    lock.unlock();

    _postsink.invalidate_samples( Signal::Interval( offset, offset + framesPerBuffer ));

    return paContinue;
}

} // namespace Adapters
