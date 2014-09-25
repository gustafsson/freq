#include "qtmicrophone.h"
#include "log.h"
#include "cpumemorystorage.h"
#include "heightmap/uncaughtexception.h"

#include <QAudioInput>

GotData::GotData(
        shared_state<Signal::Recorder::Data> data,
        Signal::Recorder::IGotDataCallback::ptr& invalidator,
        QAudioFormat format,
        QObject* parent)
    :
      QIODevice(parent),
      data(data),
      invalidator(invalidator),
      format(format)
{
    open(QIODevice::WriteOnly);
}


GotData::~GotData() {
    close();
}


qint64 GotData::
        readData(char *data, qint64 maxlen)
{
    Q_UNUSED(data)
    Q_UNUSED(maxlen)

    return 0;
}


template<class T>
void makefloat(const char* data, qint64 len, std::vector<float>& f)
{
    T* p = (T*)data;
    qint64 tlen = len / sizeof(T);
    f.resize (tlen);
    float minval = std::numeric_limits<T>::min ();
    float maxval = std::numeric_limits<T>::max ();
    for (qint64 i=0; i < tlen; i++)
        f[i] = -1.f + 2.f*(p[i] - minval)/(1.f + maxval - minval);
}

qint64 GotData::
        writeData(const char *in, qint64 len)
{
    switch (format.sampleType ())
    {
    case QAudioFormat::Float:
        writeData((const float*)in, len/format.channelCount ()/sizeof(float));
        break;
    case QAudioFormat::SignedInt:
    case QAudioFormat::UnSignedInt:
        switch (format.sampleSize ())
        {
        case 8:
            if (format.sampleType () == QAudioFormat::SignedInt) makefloat<qint8>(in, len, f);
            if (format.sampleType () == QAudioFormat::UnSignedInt) makefloat<quint8>(in, len, f);
            break;
        case 16:
            if (format.sampleType () == QAudioFormat::SignedInt) makefloat<qint16>(in, len, f);
            if (format.sampleType () == QAudioFormat::UnSignedInt) makefloat<quint16>(in, len, f);
            break;
        default:
            Log("Unknown sample size %d") % format.sampleSize ();
            return 0;
        }
        writeData(&f[0], f.size ());
        break;
    default:
        Log("Unknown sample type %d") % format.sampleType ();
        return 0;
    }

    return len;
}


void GotData::
       writeData(const float* in, quint64 len)
{
    // Prepare buffer
    Signal::IntervalType number_of_samples = len / data.raw()->num_channels;
    if (!buffer || buffer->number_of_samples () != number_of_samples)
        buffer.reset (new Signal::Buffer(0, number_of_samples, data.raw()->sample_rate, data.raw()->num_channels));

    // get pointers to intermediate buffers
    unsigned k = 0;
    unsigned C = buffer->number_of_channels ();
    float* P[C], v[C];
    bool prev_nonzero = true;

    for (unsigned i=0; i<C; ++i)
    {
        Signal::pMonoBuffer b = buffer->getChannel (i);
        P[i] = CpuMemoryStorage::WriteAll<1>(b->waveform_data()).ptr ();
    }

    // transpose and skip zeros
    for (unsigned j=0; j<number_of_samples; ++j)
    {
        bool nonzero = false;
        for (unsigned i=0; i<C; ++i)
        {
            v[i] = in[j*C + i];
            if (v[i] != 0.f)
                nonzero = true;
        }

        if (k==1 && !nonzero)
            k = 0;

        if (nonzero || prev_nonzero)
        {
            for (unsigned i=0; i<C; ++i)
                P[i][k] = v[i];
            k++;
        }

        prev_nonzero = nonzero;
    }

    if (k<=1)
        return;

    // Publish
    auto dw = data.write ();
    Signal::IntervalType start = dw->samples.spannedInterval().last;
    buffer->set_sample_offset (start);
    dw->samples.put( buffer ); // copy

    // Ignore samples not written to
    Signal::Interval I(start, start+k);
    dw->samples.invalidate_samples( Signal::Interval(I.last, Signal::Interval::IntervalType_MAX));
    dw.unlock ();

    if (invalidator) invalidator.write ()->markNewlyRecordedData( I );
}


QtMicrophone::
        QtMicrophone()
{
    qRegisterMetaType<QAudio::State>("QAudio::State");
    QAudioFormat format;
    // Set up the desired format, for example:
    format.setSampleRate (44100);
//    format.setSampleType (QAudioFormat::Float);
    format.setChannelCount (1);
//    format.setChannelCount (2);

    QAudioDeviceInfo info = QAudioDeviceInfo::defaultInputDevice();
    if (!info.isFormatSupported(format))
        format = info.nearestFormat(format);

    audio_.reset (new QAudioInput(info, format));
    audio_->setBufferSize (1<<14); // buffer_size/sample_rate = latency
    audio_->setBufferSize (512); // buffer_size/sample_rate = latency

    auto e = audio_->error ();
    if (e != QAudio::NoError)
        Log("QtMicrophone error: ") % (int)e;

    _data.reset (new Recorder::Data(format.sampleRate (), format.channelCount ()));
    device_.reset (new GotData(_data, _invalidator, format));
}


QtMicrophone::
        ~QtMicrophone()
{
    // Drop any recorded data not already processed (don't stopRecording()
    // as it would invoke GotData::writeData which uses invalidator which
    // in turn uses the dag. But the Dag is locked by the caller of this
    // destructor.
    audio_->reset();

    // Release resources
    audio_.reset ();
    device_.reset ();
}


void QtMicrophone::
        startRecording()
{
    audio_->start (device_.data ());

    if (audio_->error () != QAudio::NoError)
    {
        Log("QtMicrophone::startRecording failed: %d") % audio_->error ();
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
    switch(audio_->state ())
    {
    case QAudio::StoppedState:
    case QAudio::SuspendedState:
        return true;
    default:
        return false;
    }
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
