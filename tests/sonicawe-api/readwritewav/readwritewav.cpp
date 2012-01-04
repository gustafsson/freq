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
    try
    {
    TaskTimer tt("%s", __FUNCTION__);
    if (!QFile::exists(source.c_str()))
    {
        QFAIL("You need to run the script source.m to create some source data first");
    }

    Signal::pOperation audiofile(new Adapters::Audiofile(source));
    {
        TaskTimer t2("Writing audiofile '%s' while copying source '%s'",
                 output.c_str(), source.c_str());

        Signal::pOperation wavwrite(new Adapters::WriteWav(output));
        wavwrite->source(audiofile);
        wavwrite->invalidate_samples(wavwrite->getInterval());

        wavwrite->readFixedLength(audiofile->getInterval());
    }

    TaskTimer t2("Reading output '%s' and verifying against original input '%s'",
             output.c_str(), source.c_str());

    Signal::pOperation audiofile2(new Adapters::Audiofile(output));

    QCOMPARE( audiofile2->getInterval(), audiofile->getInterval() );

    Signal::pBuffer b = audiofile->readFixedLengthAllChannels(audiofile->getInterval());
    Signal::pBuffer b2 = audiofile->readFixedLengthAllChannels(audiofile->getInterval());

    QCOMPARE( b->waveform_data()->numberOfBytes(), b2->waveform_data()->numberOfBytes() );
    int bufferdiff = memcmp(b->waveform_data()->getCpuMemory(), b2->waveform_data()->getCpuMemory(), b2->waveform_data()->numberOfBytes() );
    QVERIFY( 0 == bufferdiff );
    }
    catch (std::exception &x)
    {
        TaskInfo("%s, %s caught %s: %s",
                 vartype(*this).c_str(), __FUNCTION__,
                 vartype(x).c_str(), x.what());
        throw;
    }
}


void ReadWriteWav::
        writeNormalized()
{
    try
    {
    TaskTimer ti("ReadWriteWav::writeNormalized");
    Signal::pOperation audiofile(new Adapters::Audiofile(source));
    if (!QFile::exists(source.c_str()))
    {
        QFAIL("You need to run the script source.m to create some source data first");
    }

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
        std::string goldname = w->normalize() ? normalizedGold : source;
        Signal::pOperation normalizedAudiofileGold(new Adapters::Audiofile(goldname));
        if (!QFile::exists(normalizedOutput.c_str()))
        {
            QFAIL(QString("Couldn't write '%1'")
                  .arg(normalizedOutput.c_str()).toStdString().c_str());
        }
        if (!QFile::exists(goldname.c_str()))
        {
            QFAIL(QString("You need to validate the output of the previous test manually and rename '%1' to '%2'")
                  .arg(normalizedOutput.c_str())
                  .arg(goldname.c_str()).toStdString().c_str());
        }

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
    catch (std::exception &x)
    {
        TaskInfo("%s, %s caught %s: %s",
                 vartype(*this).c_str(), __FUNCTION__,
                 vartype(x).c_str(), x.what());
        throw;
    }
}


QTEST_MAIN(ReadWriteWav);
#include "readwritewav.moc"
