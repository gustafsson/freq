#include "sawe/project_header.h"
#include <QtCore/QString>
#include <QtTest/QtTest>
#include <QtCore/QCoreApplication>
#include <iostream>
#include <QGLWidget> // libsonicawe uses gl, so we need to include a gl header in this project as well

#include "tfr/stft.h"

using namespace std;

class StftTest : public QObject
{
    Q_OBJECT

public:
    StftTest();

private slots:
    void initTestCase();
    void cleanupTestCase();
    void equalForwardInverse_data();
    void equalForwardInverse();

private:
    Signal::pBuffer b;
    bool coutinfo;
    int N, windowsize, ffts;

    float epsilon[2];
    std::vector<float> diffs, forwardtime, inversetime;

    unsigned passedTests;
};


StftTest::StftTest()
{
    coutinfo = false;
    //int N = 1<<23, windowsize=1<<16;
    N = 1<<22;
    windowsize=1<<10;

    epsilon[0] = 2e-7 * log((double)N);
    epsilon[1] = 2e-7 * log((double)windowsize);

    ffts = 2;
    diffs.resize(ffts*2);
    forwardtime.resize(ffts*2);
    inversetime.resize(ffts*2);

    passedTests = 0;
}

void StftTest::initTestCase()
{
    TaskTimer tt("Initiating test signal");
    b.reset(new Signal::Buffer(0, N, 1));

    float* p = b->waveform_data()->getCpuMemory();
    srand(0);
    for (int i=0; i<N; ++i)
        p[i] = 2.f*rand()/RAND_MAX - 1.f;

    if (coutinfo) cout << "buffer" << endl;
    if (coutinfo) for (int i=0; i<N; ++i)
    {
        cout << i << ", " << p[i] << ";" << endl;
    }
    if (coutinfo) cout << endl;

    {
        TaskTimer tt("Warming up...");

        Tfr::Stft a;
        a.set_approximate_chunk_size(windowsize);
        a(b);

        Tfr::Fft f;
        f(b);
    }
}


void StftTest::cleanupTestCase()
{
    if (coutinfo) cout << endl;
    for (int i=0; i<ffts*2; ++i)
        cout << (i/2?"Stft":"Fft") << " " << (i%2?"C2C":"R2C") << " " << diffs[i] << " < " << epsilon[i/2] << " " << (diffs[i]<epsilon[i/2]?"ok":"failed") << ". Time: " << forwardtime[i] << " s and " << inversetime[i] << " s. Sum " << forwardtime[i]+inversetime[i] << " s" << endl;

    cout << "Passed tests: " << passedTests << endl;
}


void StftTest::
        equalForwardInverse_data()
{
    QTest::addColumn<int>("stft");
    QTest::addColumn<bool>("redundant");

    for (int stft=0; stft<ffts; ++stft)
    {
        for (int redundant=0; redundant<2; ++redundant)
        {
            QTest::newRow(QString("%1:%2").arg(stft?"Stft":"Fft").arg(redundant?"C2C":"R2C").toLocal8Bit().data()) << stft << (bool)redundant;
        }
    }
}


void StftTest::
    equalForwardInverse()
{
    QFETCH(int, stft);
    QFETCH(bool, redundant);

    QBENCHMARK
    {

    float* p = b->waveform_data()->getCpuMemory();

    float norm = 1.f/N;
    Tfr::pTransform ft;
    if (stft)
    {
        norm = 1.f;
        Tfr::Stft* stft;
        ft.reset(stft = new Tfr::Stft);
        stft->set_approximate_chunk_size(windowsize);
        stft->compute_redundant(redundant);
    }
    else
    {
        ft.reset( new Tfr::Fft(redundant) );
    }

    Tfr::pChunk c;
    b->waveform_data()->getCpuMemory();
    {
        TaskTimer tt("%s %s forward", stft?"Stft":"Fft", redundant?"C2C":"R2C", 0);
        c = (*ft)(b);
        forwardtime[stft*2+redundant]=tt.elapsedTime();
    }

    Signal::pBuffer b2;
    {
        TaskTimer tt("%s %s backward", stft?"Stft":"Fft", redundant?"C2C":"R2C", 0);
        b2 = ft->inverse(c);
        inversetime[stft*2+redundant]=tt.elapsedTime();
    }

    std::complex<float>* cp = c->transform_data->getCpuMemory();
    if (coutinfo) cout << vartype(*ft).c_str() << " " << (redundant?"C2C":"R2C") << endl;

    if (coutinfo) for (unsigned i=0; i<c->transform_data->numberOfElements(); ++i)
        cout << i << ", " << cp[i].real() << ", " << cp[i].imag() << ";" << endl;

    float* p2 = b2->waveform_data()->getCpuMemory();

    if (coutinfo) cout << "inverse" << endl;
    float ft_diff = 0;
    for (int i=0; i<N; ++i)
    {
        if (coutinfo) cout << i << ", " << p[i] << ", " << p2[i]*norm <<  ";";
        float diff = p[i] - p2[i]*norm;
        if (coutinfo) if (fabsf(diff) > epsilon[stft])
            cout << " Failed: " << diff;
        if (ft_diff < fabsf(diff))
            ft_diff = fabsf(diff);
        if (coutinfo) cout << endl;
    }

    diffs[stft*2+redundant] = ft_diff;
    if (coutinfo) cout << endl;

    passedTests += ft_diff < epsilon[stft];

    QVERIFY( ft_diff < epsilon[stft] );

    }
}

QTEST_MAIN(StftTest);
#include "stfttest.moc"
