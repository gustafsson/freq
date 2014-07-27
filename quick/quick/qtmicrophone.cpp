#include "qtmicrophone.h"
#include "log.h"
#include "cpumemorystorage.h"

#include <QAudioInput>

QtMicrophone::
        QtMicrophone()
{
    QAudioFormat format;
    // Set up the desired format, for example:
    format.setSampleRate(44100);
    format.setSampleType(QAudioFormat::Float);
    audio_.reset (new QAudioInput(format));

    auto e = audio_->error ();
    if (e != QAudio::NoError)
        Log("QtMicrophone error: ") % (int)e;

    this->connect (audio_.data (), SIGNAL(notify()), SLOT(gotData()));
}


void QtMicrophone::
        startRecording()
{
    device_ = audio_->start ();
    if (audio_->error ())
    {
        device_ = 0;
        return;
    }

    _offset = length();
    _start_recording.restart ();
}


void QtMicrophone::
        stopRecording()
{
    audio_->stop ();
}


bool QtMicrophone::
        isStopped() const
{
    return audio_->state () != QAudio::ActiveState;
}


bool QtMicrophone::
        canRecord()
{
    return QAudio::NoError == audio_->error ();
}


std::string QtMicrophone::
        name()
{
    return "QtMicrophone";
}


void QtMicrophone::
        gotData()
{
    // What data did we got:
    Signal::Interval I;
    I.first = this->number_of_samples ();
    I.last = I.first + device_->bytesAvailable ()/audio_->format ().bytesPerFrame ();

    // Prepare buffers to receive data
    if (!buffer_ || buffer_->number_of_samples () != (Signal::IntervalType)I.count ())
    {
        buffer_.reset (new Signal::Buffer(0, I.count (), audio_->format ().sampleRate (), audio_->format ().channelCount ()));
        samples_.resize (I.count () * buffer_->number_of_channels ());
    }

    // Read data
    device_->read ((char*)&samples_[0], audio_->format ().bytesForFrames (I.count ()));

    // Transpose
    const float *in = &samples_[0];
    unsigned C = buffer_->number_of_channels ();
    for (unsigned i=0; i<C; ++i)
    {
        Signal::pMonoBuffer b = buffer_->getChannel (i);
        float* p = CpuMemoryStorage::WriteAll<1>(b->waveform_data()).ptr ();
        for (unsigned j=0; j<I.count (); ++j)
            p[j] = in[j*C + i];
    }

    _data.write ()->samples.put( buffer_ );

    if (_invalidator)
        // Tell someone that there is new data available to read
        _invalidator.write ()->markNewlyRecordedData( I );
}
