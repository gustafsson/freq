#include "sawe/project_header.h"
#include <QtCore/QString>
#include <QtTest/QtTest>
#include <QtCore/QCoreApplication>
#include <iostream>
#include <QGLWidget> // libsonicawe uses gl, so we need to include a gl header in this project as well

#include "adapters/audiofile.h"
#include "adapters/writewav.h"

using namespace std;

class ReadWriteWav : public QObject
{
    Q_OBJECT

public:
    ReadWriteWav();

private slots:
    void initTestCase();
    void cleanupTestCase();

    void readWriteWav();
    void writeNormalized();

private:
    std::string source;
    std::string output;
    std::string normalizedOutput;
    std::string normalizedGold;
};


ReadWriteWav::
        ReadWriteWav()
{
    source = "source.wav";
    output = "output.wav";
    normalizedOutput = "normalized-result.wav";
    normalizedGold = "normalized-gold.wav";
}


void ReadWriteWav::
        initTestCase()
{
}


void ReadWriteWav::
        cleanupTestCase()
{
}


void ReadWriteWav::
        readWriteWav()
{
    Signal::pOperation audiofile(new Adapters::Audiofile(source));
    {
        Signal::pOperation wavwrite(new Adapters::WriteWav(output));
        wavwrite->source(audiofile);
        wavwrite->invalidate_samples(wavwrite->getInterval());

        wavwrite->readFixedLength(audiofile->getInterval());
    }

    Signal::pOperation audiofile2(new Adapters::Audiofile(output));

    QCOMPARE( audiofile2->getInterval(), audiofile->getInterval() );

    Signal::pBuffer b = audiofile->readFixedLengthAllChannels(audiofile->getInterval());
    Signal::pBuffer b2 = audiofile->readFixedLengthAllChannels(audiofile->getInterval());

    QCOMPARE( b->waveform_data()->numberOfBytes(), b2->waveform_data()->numberOfBytes() );
    int bufferdiff = memcmp(b->waveform_data()->getCpuMemory(), b2->waveform_data()->getCpuMemory(), b2->waveform_data()->numberOfBytes() );
    QVERIFY( 0 == bufferdiff );
}


void ReadWriteWav::
        writeNormalized()
{
    Signal::pOperation audiofile(new Adapters::Audiofile(source));

    Adapters::WriteWav* w = 0;
    Signal::pOperation wavwrite(w = new Adapters::WriteWav(normalizedOutput));
    wavwrite->source(audiofile);
    wavwrite->invalidate_samples(audiofile->getInterval());
    for (int i=0; i<4; ++i)
    {
        TaskTimer ti("ReadWriteWav::writeNormalized i=%d", i);

        w->normalize(0 == i%2);

        wavwrite->readFixedLength(audiofile->getInterval());

        Signal::pOperation normalizedAudiofile(new Adapters::Audiofile(normalizedOutput));
        Signal::pOperation normalizedAudiofileGold(new Adapters::Audiofile(w->normalize() ? normalizedGold : source));

        QCOMPARE( normalizedAudiofile->getInterval(), audiofile->getInterval() );
        QCOMPARE( normalizedAudiofile->getInterval(), normalizedAudiofileGold->getInterval() );

        Signal::pBuffer b = normalizedAudiofile->readFixedLengthAllChannels(audiofile->getInterval());
        Signal::pBuffer b2 = normalizedAudiofileGold->readFixedLengthAllChannels(audiofile->getInterval());

        TaskTimer t2("ReadWriteWav::writeNormalized i=%d", i);

        QCOMPARE( b->waveform_data()->numberOfBytes(), b2->waveform_data()->numberOfBytes() );
        float maxdiff = 0;

        float *p = b->waveform_data()->getCpuMemory();
        float *p2 = b2->waveform_data()->getCpuMemory();
        for (unsigned x=0; x<b->waveform_data()->numberOfElements(); ++x )
        {
            float& v = p[x];
            float& v2 = p2[x];
            maxdiff = std::max( maxdiff, std::fabs(v - v2) );
        }

        if (maxdiff > 3e-4)
            QCOMPARE( maxdiff, 0.f );
    }
}


QTEST_MAIN(ReadWriteWav);
#include "readwritewav.moc"
