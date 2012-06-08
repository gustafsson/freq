#include "sawe/project_header.h"
#include <QtCore/QString>
#include <QtTest/QtTest>
#include <QtCore/QCoreApplication>
#include <iostream>
#include <QGLWidget> // libsonicawe uses gl, so we need to include a gl header in this project as well

#include "tfr/stft.h"

using namespace std;
using namespace Tfr;
using namespace Signal;

namespace QTest {
template<>
char * toString ( const Signal::Interval & value )
{
    return qstrdup( value.toString().c_str() );
}
}

class StftTest : public QObject
{
    Q_OBJECT

public:
    StftTest();

private slots:
    void initTestCase();
    void cleanupTestCase();
    void compareToGold_data();
    void compareToGold();
    void equalForwardInverse_data();
    void equalForwardInverse();

private:
    float testTransform(pTransform t, pBuffer b, float* forwardtime=0, float* inversetime=0, float* epsilon=0);

    pBuffer data;
    unsigned coutinfo;
    bool benchmark_summary;
    int N, windowsize, ftruns;

    std::vector<float> diffs, forwardtime, inversetime, epsilon;
    float overlap;

    unsigned passedTests, failedTests;
    pBuffer gold_input_data;
    std::vector<std::complex<float> > gold_ft;
    float gold_overlap;
    Stft::WindowType gold_window;
    int gold_windowsize;
};


StftTest::StftTest()
{
    coutinfo = 60;
    benchmark_summary = false;
    //N = 1<<23; windowsize=1<<16;
    //N = 1<<22; windowsize=1<<10;
    N = 16; windowsize=4;
    N = 1024; windowsize=256;
    //N = 64; windowsize=16;
    overlap = 0.75f;

    epsilon.resize(1+Stft::WindowType_NumberOfWindowTypes);
    // these are given for windowsize 256 and overlap 0.75
    epsilon[0] = 2e-7 * log((double)N);
    epsilon[1+Stft::WindowType_Rectangular] = 2e-7 * log((double)windowsize); // rectangular
    epsilon[1+Stft::WindowType_Hann] = .0002f;
    epsilon[1+Stft::WindowType_Hamming] = .00005f;
    epsilon[1+Stft::WindowType_Tukey] = .004f;
    epsilon[1+Stft::WindowType_Cosine] = .001f;
    epsilon[1+Stft::WindowType_Lanczos] = .003f;
    epsilon[1+Stft::WindowType_Triangular] = 2e-7 * log((double)windowsize);
    epsilon[1+Stft::WindowType_Gaussian] = .004f;
    epsilon[1+Stft::WindowType_BarlettHann] = .0006f;
    epsilon[1+Stft::WindowType_Blackman] = .0003f;
    epsilon[1+Stft::WindowType_Nuttail] = 3e-5;
    epsilon[1+Stft::WindowType_BlackmanHarris] = 3e-5;
    epsilon[1+Stft::WindowType_BlackmanNuttail] = 4e-5;
    epsilon[1+Stft::WindowType_FlatTop] = 0.03f;

#ifdef __APPLE__
    // we haven't investigated why the accuracy differs
    epsilon[1+Stft::WindowType_Hamming] = .00006f;
    epsilon[1+Stft::WindowType_FlatTop] = 0.035f;
#endif

    ftruns = 1 + Stft::WindowType_NumberOfWindowTypes;
    diffs.resize(ftruns*2);
    forwardtime.resize(ftruns*2);
    inversetime.resize(ftruns*2);

    passedTests = 0;
    failedTests = 0;
}

void StftTest::initTestCase()
{
    TaskTimer tt("Initiating test signal");
    data.reset(new Buffer(0, N, 1));

    float* p = data->waveform_data()->getCpuMemory();
    srand(0);
    for (int i=0; i<N; ++i)
        p[i] = 2.f*rand()/RAND_MAX - 1.f;

    gold_input_data.reset( new Buffer(4,16,1));
    float *gold_input = gold_input_data->waveform_data()->getCpuMemory();
    gold_overlap = 0.5;
    gold_windowsize = 4;
    gold_ft.resize( 7*gold_windowsize );
    gold_window = Stft::WindowType_Rectangular;
    gold_input[0]=.4072680;
    gold_input[1]=.5475688;
    gold_input[2]=.8947876;
    gold_input[3]=.4636770;
    gold_input[4]=.3359026;
    gold_input[5]=.0809182;
    gold_input[6]=.2194318;
    gold_input[7]=.0091178;
    gold_input[8]=.7669713;
    gold_input[9]=.4394998;
    gold_input[10]=.8835012;
    gold_input[11]=.9390582;
    gold_input[12]=.2508032;
    gold_input[13]=.0341413;
    gold_input[14]=.2510600;
    gold_input[15]=.6913113;

    gold_ft[0]=ChunkElement(2.3133,0);
    gold_ft[1]=ChunkElement(-0.48752,-0.083892);
    gold_ft[2]=ChunkElement(0.29081,0);
    gold_ft[3]=ChunkElement(-0.48752,0.083892);
    gold_ft[4]=ChunkElement(1.7753,0);
    gold_ft[5]=ChunkElement(0.55888,-0.38276);
    gold_ft[6]=ChunkElement(0.6861,0);
    gold_ft[7]=ChunkElement(0.55888,0.38276);
    gold_ft[8]=ChunkElement(0.64537,0);
    gold_ft[9]=ChunkElement(0.11647,-0.0718);
    gold_ft[10]=ChunkElement(0.4653,0);
    gold_ft[11]=ChunkElement(0.11647,0.0718);
    gold_ft[12]=ChunkElement(1.435,0);
    gold_ft[13]=ChunkElement(-0.54754,0.43038);
    gold_ft[14]=ChunkElement(0.53779,0);
    gold_ft[15]=ChunkElement(-0.54754,-0.43038);
    gold_ft[16]=ChunkElement(3.029,0);
    gold_ft[17]=ChunkElement(-0.11653,0.49956);
    gold_ft[18]=ChunkElement(0.27191,0);
    gold_ft[19]=ChunkElement(-0.11653,-0.49956);
    gold_ft[20]=ChunkElement(2.1075,0);
    gold_ft[21]=ChunkElement(0.6327,-0.90492);
    gold_ft[22]=ChunkElement(0.1611,0);
    gold_ft[23]=ChunkElement(0.6327,0.90492);
    gold_ft[24]=ChunkElement(1.2273,0);
    gold_ft[25]=ChunkElement(-0.00025684,0.65717);
    gold_ft[26]=ChunkElement(-0.22359,0);
    gold_ft[27]=ChunkElement(-0.00025684,-0.65717);

    if (coutinfo)
    {
        cout << "buffer" << endl;
        for (int i=0; i<N && i<(int)coutinfo; ++i)
        {
            cout << i << ", " << p[i] << ";" << endl;
        }
        cout << endl;

        TaskTimer tt("Warming up...");

        Stft a;
        a.set_approximate_chunk_size(windowsize);
        a(data);

        Fft f;
        f(data);
    }
}


void StftTest::cleanupTestCase()
{
    if (coutinfo)
        cout << endl;

    for (int i=0; i<2*ftruns; ++i)
    {
        bool success = diffs[i]<epsilon[i/2];
        cout << (success?"Success: ":"Failed:  ")
             << i << " " << (i==0||i==1?"Fft":("Stft " + Stft::windowTypeName((Stft::WindowType)(i/2-1))).c_str())
             << " " << (i%2?"C2C":"R2C")
             << " " << diffs[i];

        if (i<4)
            cout << " < " << epsilon[i/2];

        if (benchmark_summary)
            cout << ". Time: " << forwardtime[i] << " s and inverse " << inversetime[i]
                 << " s. Sum " << forwardtime[i]+inversetime[i] << " s";

        cout << endl;
    }

    cout << "Passed tests: " << passedTests << endl;
}



void StftTest::
        compareToGold_data()
{
    QTest::addColumn<bool>("overlap");

    for (int overlap=0; overlap<2; ++overlap)
    {
        QTest::newRow(QString("Gold %1")
                      .arg(overlap?"overlap":"no overlap").toLocal8Bit().data())
                << (bool)overlap;
    }
}

void StftTest::
    compareToGold()
{
    QFETCH(bool, overlap);

    Stft* t;
    pTransform ft(t = new Stft);
    t->setWindow( gold_window, overlap?gold_overlap:0 );
    t->set_approximate_chunk_size( gold_windowsize );
    t->compute_redundant( true );

    float ft_diff = testTransform( ft, gold_input_data, 0, 0, coutinfo?&epsilon[1]:0);

    if (coutinfo && ft_diff>epsilon[1])
    {
        pChunk c = (*t)(gold_input_data);
        if (overlap)
            QCOMPARE( c->getInterval(), gold_input_data->getInterval() );

        std::complex<float>* cp = c->transform_data->getCpuMemory();
        gold_ft.resize( 7*gold_windowsize );

        std::vector<std::complex<float> > gold_ft_nonredundant;
        std::vector<std::complex<float> >* gold_ft_data = 0;
        if (overlap)
            gold_ft_data = &gold_ft;
        else
        {
            gold_ft_nonredundant.resize(16);

            for (int i=0; i<4; ++i)
                for (int j=0; j<4; ++j)
                    gold_ft_nonredundant[i*4+j] = gold_ft[i*8+j];

            gold_ft_data = &gold_ft_nonredundant;
        }

        std::complex<float>* gold_cp = &(*gold_ft_data)[0];
        QCOMPARE( c->transform_data->numberOfElements(), gold_ft_data->size() );

        cout << "Gold transform\t\tComputed transform" << endl;
        for (unsigned i=0; i<c->transform_data->numberOfElements() && i<coutinfo; ++i)
        {
            cout << i << ", " << gold_cp[i].real() << ",\t" << gold_cp[i].imag() << ";" << endl;
            cout << i << ", " << cp[i].real() << ",\t" << cp[i].imag() << ";" << endl;
        }

        pBuffer b2 = t->inverse(c);
    }

    passedTests += ft_diff < epsilon[1];

    QVERIFY( ft_diff < epsilon[1] );
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
                          .arg(stft?Stft::windowTypeName((Stft::WindowType)(stft-1)).c_str():"")
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
        pTransform ft;
        if (stft)
        {
            Stft* t;
            ft.reset(t = new Stft);
            t->setWindow( (Stft::WindowType)(stft-1), overlap );
            t->set_approximate_chunk_size(windowsize);
            t->compute_redundant(redundant);
        }
        else
        {
            ft.reset( new Fft(redundant) );
        }

        float ft_diff = testTransform( ft, data, &forwardtime[stft*2+redundant], &inversetime[stft*2+redundant], coutinfo?&epsilon[stft]:0);
        diffs[stft*2+redundant] = ft_diff;

        bool success = ft_diff < epsilon[stft];
        passedTests += success;
        failedTests += !success;

        QVERIFY( ft_diff < epsilon[stft] );
    }
}


float StftTest::
    testTransform(pTransform t, pBuffer b, float* forwardtime, float* inversetime, float *epsilon)
{
    float norm = 1.f;
    bool isfft = 0!=dynamic_cast<Fft*>(t.get());
    if (isfft)
        norm = 1.f/b->number_of_samples();

    pChunk c;
    b->waveform_data()->getCpuMemory();
    {
        TaskTimer tt("%s forward", t->toString().c_str());
        c = (*t)(b);
        if (forwardtime) *forwardtime=tt.elapsedTime();
    }

    pBuffer b2;
    {
        TaskTimer tt("%s backward", t->toString().c_str());
        b2 = t->inverse(c);
        if (inversetime) *inversetime=tt.elapsedTime();
    }


    float ft_diff = 0;
    Interval b2i = b2->getInterval();
    pBuffer expected = BufferSource(b).readFixedLength(b2i);
    float *expectedp = expected->waveform_data()->getCpuMemory();
    float* p2 = b2->waveform_data()->getCpuMemory();

    unsigned start = 0;
    if (!isfft && 0 == b2i.first)
    {
        StftChunk* stftchunk = dynamic_cast<StftChunk*>(c.get());
        start = stftchunk->window_size() - stftchunk->increment();
    }

    for (unsigned i=start; i<b2->number_of_samples(); ++i)
    {
        float diff = fabsf(expectedp[i] - p2[i]*norm);
        if (ft_diff < diff)
            ft_diff = diff;
    }

    if (epsilon && ft_diff>*epsilon)
    {
        cout << t->toString() << endl;

        std::complex<float>* cp = c->transform_data->getCpuMemory();
        for (unsigned i=0; i<c->transform_data->numberOfElements() && i<coutinfo; ++i)
            cout << i << ", " << cp[i].real() << ", " << cp[i].imag() << ";" << endl;

        cout << "expected, inverse, diff, frac" << endl;

        for (unsigned i=start; i<b2->number_of_samples()&&i<coutinfo; ++i)
        {
            float diff = expectedp[i] - p2[i]*norm;
            float frac = expectedp[i] / p2[i]*norm;
            cout << (b2->sample_offset+i).asFloat() << ", " << expectedp[i] << ", " << p2[i]*norm <<  ", " << diff << ", " << frac<< ";" << endl;
        }
    }

    return ft_diff;
}


QTEST_MAIN(StftTest);
#include "stfttest.moc"
