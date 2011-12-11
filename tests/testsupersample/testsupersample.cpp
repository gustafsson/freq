#include "sawe/project_header.h"
#include <QtCore/QString>
#include <QtTest/QtTest>
#include <QtCore/QCoreApplication>
#include <iostream>
#include <QGLWidget> // libsonicawe uses gl, so we need to include a gl header in this project as well

#include "tfr/supersample.h"

using namespace std;

class TestSuperSample : public QObject
{
    Q_OBJECT

public:
    TestSuperSample();

private slots:
    void initTestCase();
    void cleanupTestCase();
    void superSampleInterpolation();
    void superSampleInterpolation_data();

private:
    Signal::pBuffer data, precalced, expectedb2;
    int data_N;

    float epsilon;
    int coutinfo;
    int passedConfigurations;

    std::vector<float> diffs;
};


TestSuperSample::TestSuperSample()
{
    data_N = 1<<22;
    data_N = 256;
    coutinfo = 0;

    epsilon = 2e-7 * log((double)data_N);

    passedConfigurations = 0;
}


void TestSuperSample::initTestCase()
{
    TaskTimer tt("Initiating test signal");
    data.reset(new Signal::Buffer(data_N, data_N, 1));

    float* p = data->waveform_data()->getCpuMemory();
    srand(0);
    for (int i=0; i<data_N; ++i)
        p[i] = 2.f*rand()/RAND_MAX - 1.f;

    precalced.reset(new Signal::Buffer(0, 4, 1));
    float* r = precalced->waveform_data()->getCpuMemory();
    r[0] = -0.72730;
    r[1] = -0.28839;
    r[2] = 0.63661;
    r[3] = 0.57604;

    expectedb2.reset(new Signal::Buffer(0, 2*4, 1));
    float* q = expectedb2->waveform_data()->getCpuMemory();
    q[0] = -0.72730;
    q[1] = -0.73860;
    q[2] = -0.28839;
    q[3] = 0.22583;
    q[4] = 0.63661;
    q[5] = 0.83708;
    q[6] = 0.57604;
    q[7] = -0.12735;

    // note r[i]==q[i*2]

    if (coutinfo)
    {
        cout << endl;
        for (int i=0; i<data_N && i<coutinfo; ++i)
            cout << i << ", " << p[i] << ";" << endl;
    }
}


void TestSuperSample::cleanupTestCase()
{
    cout << "Passed configurations: " << passedConfigurations << endl;

    for (unsigned i=0; i<diffs.size(); ++i)
    {
        if (diffs[i]>=epsilon)
            cout << i << ": " << diffs[i] << endl;
    }
}


void TestSuperSample::
        superSampleInterpolation_data()
{
    QTest::addColumn<int>("multiple");

    for (int multiple=0; multiple<10; ++multiple)
    {
        QString info;
        if (multiple==0)
            info = "compare to gold";
        else
            info = QString("multiple = %1").arg(1<<multiple);

        QTest::newRow(info.toLocal8Bit().data()) << (1<<multiple);
    }
}

void TestSuperSample::
    superSampleInterpolation()
{
    QFETCH(int, multiple);

    Signal::pBuffer b;
    if (1 == multiple)
    {
        b = precalced;
        multiple = 2;
    }
    else
        b = data;

    int N = b->waveform_data()->numberOfElements();

    Signal::pBuffer super = Tfr::SuperSample::supersample(b, b->sample_rate*multiple);

    QCOMPARE( (int)super->waveform_data()->numberOfElements(), multiple*N );
    QCOMPARE( super->sample_rate, multiple*b->sample_rate );
    QCOMPARE( super->start(), b->start() );
    QCOMPARE( super->length(), b->length() );


    float* p = b->waveform_data()->getCpuMemory();
    float* p2 = super->waveform_data()->getCpuMemory();

    float ft_diff = 0;
    for (int i=0; i<N; ++i)
    {
        float diff = fabsf(p[i] - p2[multiple*i]);
        if (ft_diff < diff)
            ft_diff = diff;
    }


    if (coutinfo)
    {
        if (b == precalced)
        {
            float* q = expectedb2->waveform_data()->getCpuMemory();
            for (int i=0; i<2*N && i<multiple*coutinfo; ++i)
                cout << i << ", " << p2[i] << ", " << q[i] <<  ";" << endl;
        }
        else
        {
            for (int i=0; i<2*N && i<multiple*coutinfo; ++i)
                cout << i << ", " << p2[i] << ";" << endl;
        }

        for (int i=0; i<N && i<coutinfo; ++i)
        {
            cout << i << ", " << p[i] << ", " << p2[multiple*i] <<  ";";
            float diff = p[i] - p2[multiple*i];
            if (fabsf(diff) > epsilon)
                cout << " Failed: " << diff;
            cout << endl;
        }
    }

    diffs.push_back( ft_diff );
    passedConfigurations += ft_diff < epsilon;
    QVERIFY( ft_diff < epsilon );
}

QTEST_MAIN(TestSuperSample);
#include "testsupersample.moc"
