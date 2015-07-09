#include "qtaudiofile.h"
#include "signal/cache.h"
#include "cpumemorystorage.h"
#include "log.h"

#include <QAudioDecoder>
#include <QEventLoop>

using namespace Signal;

template<class T>
void transpose(const pBuffer& out, const void* vin)
{
    const T* in = (const T*)vin;
    int C = out->number_of_channels ();
    DataAccessPosition_t L = out->number_of_samples ();

    for (int c=0; c<C; c++)
    {
        pMonoBuffer m = out->getChannel (c);
        float* p = CpuMemoryStorage::WriteAll<1>(m->waveform_data()).ptr ();

        for (DataAccessPosition_t s=0; s<L; s++)
            p[s] = in[s*C+c];
    }
}


class FileReader : public Operation
{
public:
    FileReader(const QAudioDecoder* decoder)
        :
          decoder(decoder)
    {
    }

    pBuffer process(pBuffer b) override
    {

        Interval i = b->getInterval();
        while (!data.samplesDesc ().contains (i) && decoder->state () == QAudioDecoder::DecodingState)
        {
            QEventLoop loop;
            loop.connect(decoder, SIGNAL(bufferAvailableChanged(bool)), &loop, SLOT(quit()));
            if (!decoder->bufferAvailable ())
                loop.exec ();
            QAudioBuffer ab = decoder->read ();
            int L = ab.frameCount ();
            int C = ab.format ().channelCount ();
            const void* d = ab.constData ();
            pBuffer sb(new Buffer(Interval(data.spannedInterval ().last, data.spannedInterval ().last + L),
                              ab.format ().sampleRate (), C));
            int s = ab.format ().sampleSize ();

            switch(ab.format ().sampleType ())
            {
            case QAudioFormat::SignedInt:
                if (8 == s) transpose<int8_t>(sb,d);
                if (16 == s) transpose<int16_t>(sb,d);
                if (32 == s) transpose<int32_t>(sb,d);
                break;
            case QAudioFormat::UnSignedInt:
                if (8 == s) transpose<uint8_t>(sb,d);
                if (16 == s) transpose<uint16_t>(sb,d);
                if (32 == s) transpose<uint32_t>(sb,d);
                break;
            case QAudioFormat::Float:
                transpose<float>(sb,d);
                break;
            case QAudioFormat::Unknown:
            default:
                break;
            }

            // write zeros after end
//            qint64 l = decoder->duration ();
//            if (i.last > l*0.001*sb->sample_rate())
//                *b |= Buffer(Interval(l,i.last),b->sample_rate(),b->number_of_channels());

            data.put (sb);
        }


        data.read (b);

        return b;
    }

    Cache data;
    const QAudioDecoder* decoder;
};


QtAudiofile::QtAudiofile(QUrl url)
    :   url(url)
{
    decoder.setSourceFilename (url.toLocalFile ());
    decoder.start ();

    if (decoder.error () != QAudioDecoder::NoError)
        Log("qtaudiofile: %s") % decoder.errorString ().toStdString ();
}


Interval QtAudiofile::
        requiredInterval( const Interval& I, Interval* expectedOutput ) const
{
    if (expectedOutput)
        *expectedOutput = I;
    return I;
}


Interval QtAudiofile::
        affectedInterval( const Interval& I ) const
{
    return I;
}


OperationDesc::ptr QtAudiofile::
        copy() const
{
    return OperationDesc::ptr();
}


Operation::ptr QtAudiofile::
        createOperation(ComputingEngine* engine) const
{
    if (engine)
        return Operation::ptr();

    return Operation::ptr(new FileReader(&decoder));
}


OperationDesc::Extent QtAudiofile::
        extent() const
{
    qint64 d = decoder.duration ();
    if (decoder.error () != QAudioDecoder::NoError || d <= 0)
        return Extent();

    QAudioFormat f = decoder.audioFormat();
    Extent x;
    x.interval = Interval(0, d);
    x.number_of_channels = f.channelCount ();
    x.sample_rate = f.sampleRate ();
    return x;
}


QString QtAudiofile::
        toString() const
{
    return QString("QtAudiofile %1%").arg(url.toString ());
}


bool QtAudiofile::
        operator==(const OperationDesc& d) const
{
    const QtAudiofile* b = dynamic_cast<const QtAudiofile*>(&d);
    return b && b->url == url;
}


void QtAudiofile::
        durationChanged(qint64 duration)
{
    this->getInvalidator ()->deprecateCache (Interval(old_duration, duration));
    old_duration = duration;
}
