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
    static float testTransform(Tfr::pTransform t, Signal::pBuffer b, float* forwardtime=0, float* inversetime=0, float* epsilon=0);

    Signal::pBuffer b;
    bool coutinfo;
    bool benchmark_summary;
    int N, windowsize, ftruns;

    float epsilon[2];
    std::vector<float> diffs, forwardtime, inversetime;
    float overlap;

    unsigned passedTests;
};


StftTest::StftTest()
{
    coutinfo = true;
    benchmark_summary = false;
    //N = 1<<23; windowsize=1<<16;
    //N = 1<<22; windowsize=1<<10;
    N = 16; windowsize=4;
    overlap = 0.98f;

    epsilon[0] = 2e-7 * log((double)N);
    epsilon[1] = 2e-7 * log((double)windowsize);

    ftruns = 1 + Tfr::Stft::WindowType_NumberOfWindowTypes;
    diffs.resize(ftruns*2);
    forwardtime.resize(ftruns*2);
    inversetime.resize(ftruns*2);

    passedTests = 0;
}

void StftTest::initTestCase()
{
    TaskTimer tt("Initiating test signal");
    b.reset(new Signal::Buffer(N, N, 1));

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
    for (int i=0; i<2*ftruns; ++i)
    {
        bool success = diffs[i]<epsilon[false != (i/2)];
        cout << (success?"Success: ":"Failed:  ")
             << i << " " << (i==0||i==1?"Fft":("Stft " + Tfr::Stft::windowTypeName((Tfr::Stft::WindowType)(i/2-1))).c_str())
             << " " << (i%2?"C2C":"R2C")
             << " " << diffs[i];

        if (i<4)
            cout << " < " << epsilon[false != (i/2)];

        if (benchmark_summary)
            cout << ". Time: " << forwardtime[i] << " s and inverse " << inversetime[i]
                 << " s. Sum " << forwardtime[i]+inversetime[i] << " s";

        cout << endl;
    }

    cout << "Passed tests: " << passedTests << endl;
}


void StftTest::
        equalForwardInverse_data()
{
    QTest::addColumn<int>("stft");
    QTest::addColumn<bool>("redundant");

    for (int stft=0; stft<ftruns; ++stft)
    {
        for (int redundant=0; redundant<2; ++redundant)
        {
            QTest::newRow(QString("%1%2 %3")
                          .arg(stft?"Stft ":"Fft")
                          .arg(stft?Tfr::Stft::windowTypeName((Tfr::Stft::WindowType)(stft-1)).c_str():"")
                          .arg(redundant?"C2C":"R2C").toLocal8Bit().data())
                    << stft << (bool)redundant;
        }
    }
}


void StftTest::
    equalForwardInverse()
{
    QFETCH(int, stft);
    QFETCH(bool, redundant);

    //QBENCHMARK
    {
        Tfr::pTransform ft;
        if (stft)
        {
            Tfr::Stft* t;
            ft.reset(t = new Tfr::Stft);
            t->setWindow( (Tfr::Stft::WindowType)(stft-1), overlap );
            t->set_approximate_chunk_size(windowsize);
            t->compute_redundant(redundant);
        }
        else
        {
            ft.reset( new Tfr::Fft(redundant) );
        }

        float ft_diff = testTransform( ft, b, &forwardtime[stft*2+redundant], &inversetime[stft*2+redundant], coutinfo?&epsilon[0!=stft]:0);
        diffs[stft*2+redundant] = ft_diff;

        passedTests += ft_diff < epsilon[0!=stft];

        QVERIFY( ft_diff < epsilon[0!=stft] );
    }
}


float StftTest::
    testTransform(Tfr::pTransform t, Signal::pBuffer b, float* forwardtime, float* inversetime, float *epsilon)
{
    float norm = 1.f;
    if (dynamic_cast<Tfr::Fft*>(t.get()))
        norm = 1.f/b->number_of_samples();

    Tfr::pChunk c;
    b->waveform_data()->getCpuMemory();
    {
        TaskTimer tt("%s forward", t->toString().c_str());
        c = (*t)(b);
        if (forwardtime) *forwardtime=tt.elapsedTime();
    }

    Signal::pBuffer b2;
    {
        TaskTimer tt("%s backward", t->toString().c_str());
        b2 = t->inverse(c);
        if (inversetime) *inversetime=tt.elapsedTime();
    }


    float ft_diff = 0;
    Signal::pBuffer expected = Signal::BufferSource(b).readFixedLength(b2->getInterval());
    float *expectedp = expected->waveform_data()->getCpuMemory();
    float* p2 = b2->waveform_data()->getCpuMemory();

    for (unsigned i=0; i<b2->number_of_samples(); ++i)
    {
        float diff = fabsf(expectedp[i] - p2[i]*norm);
        if (ft_diff < diff)
            ft_diff = diff;
    }

    if (epsilon)
    {
        cout << t->toString() << endl;

        std::complex<float>* cp = c->transform_data->getCpuMemory();
        for (unsigned i=0; i<c->transform_data->numberOfElements(); ++i)
            cout << i << ", " << cp[i].real() << ", " << cp[i].imag() << ";" << endl;

        cout << "inverse" << endl;

        for (unsigned i=0; i<b2->number_of_samples(); ++i)
        {
            cout << b2->sample_offset+i << ", " << expectedp[i] << ", " << p2[i]*norm <<  ";";
            float diff = expectedp[i] - p2[i]*norm;
            if (fabsf(diff) > *epsilon)
                cout << " Failed: " << diff;
            cout << endl;
        }
    }

    return ft_diff;
}

QTEST_MAIN(StftTest);
#include "stfttest.moc"
