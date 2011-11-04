// Sonic AWE
#include "sawe/application.h"
#include "tfr/cwt.h"
#include "sawe/reader.h"

// gpumisc
#include <redirectstdout.h>

// Qt
#include <QtGui/QMessageBox>
#include <qgl.h>
#include <QDesktopServices>
#include <QDir>


using namespace std;
using namespace boost;
using namespace Ui;
using namespace Signal;

#ifndef USE_CUDA
    static bool check_cuda( bool use_OpenGL_bindings ) {
        return true;
    }
#else

// gpumisc
#include <CudaProperties.h>
#include <CudaException.h>
#include "GpuCpuData.h"

// cuda
#include <cuda_gl_interop.h>
#include <cuda.h>

static bool check_cuda( bool use_OpenGL_bindings ) {
    stringstream ss;
    void* ptr=(void*)0;
    CudaException namedError(cudaSuccess);


    try {
        CudaProperties::getCudaDeviceProp();

        {
            // Might need cudaGLSetGLDevice later on, but it can't be called
            // until we have created an OpenGL context.
            CudaException_SAFE_CALL( cudaThreadExit() );
            if (use_OpenGL_bindings)
            {
                CudaException_SAFE_CALL( cudaGLSetGLDevice( 0 ) );
            }
            else
            {
                CudaException_SAFE_CALL( cudaSetDevice( 0 ) );
            }

            CudaException_SAFE_CALL( cudaMalloc( &ptr, 1024 ));
            CudaException_SAFE_CALL( cudaFree( ptr ));
            GpuCpuData<float> a( 0, make_cudaExtent(1024,1,1), GpuCpuVoidData::CudaGlobal );
            a.getCudaGlobal();

            size_t free=0, total=0;
            cudaMemGetInfo(&free, &total);
            TaskInfo("Cuda RAM size %g MB (of which %g MB are currently available)",
                     total/1024.f/1024, free/1024.f/1024);

            if (!use_OpenGL_bindings) if (free < total/2)
            {
                std::stringstream ss;
                ss <<
                        "There seem to be one or more other applications "
                        "currently using a lot of GPU memory. This might have "
                        "a negative performance impact on Sonic AWE." << endl
                   << endl
                   << "Total memory free to use by Sonic AWE is "
                   << (free>>20) << " MB out of a total of " << (total>>20)
                   << " MB on the GPU, "
                   << CudaProperties::getCudaDeviceProp().name << "."
                   << endl
                   << endl
                   << "If you've been using the matlab/octave integration "
                   << "and have experienced any crash, make sure you've "
                   << "cleaned up all background octave processes that may "
                   << "still be running."
                   << endl << endl
                   << "Sonic AWE will now try to start without using up too "
                   "much memory.";
                QMessageBox::information(
                        0, 
                        "A lot of GPU memory is currently being used",
                        ss.str().c_str());
            }
            return true;
        }
    } catch (const CudaException& x) {
        namedError = x;

        ss << x.what() << endl << "Cuda error code " << x.getCudaError() << endl << endl;

        ptr = 0;
    } catch (...) {
        ss << "catch (...)" << endl;
        ptr = 0;
    }
    
    // Show error messages:
    std::string nvidia_url;
#ifdef __APPLE__
    nvidia_url = "\"Developer Drivers for MacOS\" at \nhttp://www.nvidia.com/object/cuda_get.html#MacOS";
#else
    nvidia_url = "www.nvidia.com";
#endif

    stringstream msg;
    stringstream title;

    switch (namedError.getCudaError())
    {
    case cudaErrorInsufficientDriver:
        title << "Display drivers to old";
        msg << "Cuda error: " << cudaGetErrorString(cudaErrorInsufficientDriver) << endl
                << endl
                << "Sonic AWE requires you to have installed more recent display drivers from NVIDIA. "
                << "Display drivers from NVIDIA are installed on this computer but they are too old. "
                << "Please download new drivers from NVIDIA:" << endl
                << endl
                << nvidia_url << endl
                << endl
                << "Sonic AWE cannot start. Please try again with updated drivers.";
        break;
#if 3000 < CUDART_VERSION
    case cudaErrorDevicesUnavailable:
        title << "Graphics adapter (GPU) occupied";
        msg << "The NVIDIA CUDA driver couldn't start because the GPU is occupied. "
                << "Are you currently using the GPU in any other application? "
                << "If you're not intentionally using the GPU right now the driver might have been left in an inconsistent state after a previous crash. Rebooting your computer could work around this for now. "
                << "Also make sure that you have installed the latest graphics drivers." << endl
                << endl
                << endl
                << "Sonic AWE cannot start. Please try again after closing some other graphics applications.";
        break;
#endif
    default:
    {
        cerr << ss.str();
        cerr.flush();

        title << "Couldn't find CUDA, cannot start Sonic AWE";
        msg   << "Sonic AWE requires you to have installed recent display drivers from NVIDIA, and no such driver was found." << endl
                << endl
                << "Hardware requirements: You need to have one of these graphics cards from NVIDIA:" << endl
                << "   www.nvidia.com/object/cuda_gpus.html" << endl
                << endl
                << "Software requirements: You also need to have installed recent display drivers from NVIDIA:" << endl
                << endl
                << nvidia_url << endl
                << endl
                << "Sonic AWE cannot start. Please try again with updated drivers.";
    }
    }

    TaskInfo("%s\n%s", title.str().c_str(), msg.str().c_str());
    QMessageBox::critical( 0,
                 QString::fromLocal8Bit(title.str().c_str()),
                 QString::fromLocal8Bit(msg.str().c_str()) );

    return false;
}
#endif // USE_CUDA


#include "heightmap/resampletest.h"
#include "tools/support/brushpaint.cu.h"
#include "tfr/supersample.h"
#include <Statistics.h>
#include "adapters/audiofile.h"
#include "adapters/writewav.h"
//#include <boost/archive/xml_oarchive.hpp>
//#include <boost/archive/xml_iarchive.hpp>
#include <fstream>
#include <boost/algorithm/string.hpp>

using namespace Signal;

class A
{
public:
    A() { std::cout << __FUNCTION__ << " " << this << std::endl; }
    virtual ~A() { std::cout << __FUNCTION__ << " " << this << std::endl; }

    int data;
};

A hej()
{
    return A();
}

class B
{
public:
    virtual ~B() { std::cout << __FUNCTION__ << " " << this << std::endl; }
    int data2;
};

class C: public A, public B
{
public:
    virtual ~C() { std::cout << __FUNCTION__ << " " << this << std::endl; }
    int data3;
};

void tsta(A*a)
{
    std::cout << a << " a " << a->data << std::endl;
}

void tstb(B*b)
{
    std::cout << b << " b " << b->data2 << std::endl;
}
void tstc(C*c)
{
    std::cout << c << " c " << c->data3 << std::endl;
}

int main(int argc, char *argv[])
{
    QString localAppDir = QDesktopServices::storageLocation(QDesktopServices::DataLocation);
#ifdef _MSC_VER
    #if defined(TARGET_reader)
        localAppDir += "\\MuchDifferent\\Sonic AWE Reader\\";
    #else
        localAppDir += "\\MuchDifferent\\Sonic AWE\\";
    #endif
#else
    #if defined(TARGET_reader)
        localAppDir += "/MuchDifferent/Sonic AWE Reader/";
    #else
        localAppDir += "/MuchDifferent/Sonic AWE/";
    #endif
#endif
    if (QDir(localAppDir).exists()==false)
        QDir().mkpath(localAppDir);

    std::string logdir = localAppDir.toLocal8Bit().data();
    std::string logpath = logdir+"sonicawe.log";
#ifndef _MSC_VER
    //The following line hinders the redirection from working in windows
    cout << "Saving log file at \"" << logpath << "\"" << endl;
#endif

    if (1)
    {
#ifndef USE_CUDA
        std::complex<float> f(1.123456789876543, 2.9876545678903);
        cufftComplex& b = *(cufftComplex*)&f;
        cufftComplex c;
        BOOST_ASSERT( b.x == f.real() && b.y == f.imag());
        BOOST_ASSERT( sizeof(f) == sizeof(c) );
#endif
    }

    if (0)
    {
        Intervals I(403456,403457);
        Intervals J(0,403456);
        cout << ((I-J) & J) << endl;
        return 0;
    }
    if (0)
    {
        C* c = new C;
        A* a = c;
        tsta(c);
        tstb(c);
        tstc(c);
        delete a;
        c = new C;
        B* b = c;
        tsta(c);
        tstb(c);
        tstc(c);
        delete b;
        return 0;
    }
    if (0)
    {
        C c;
        c.data = 1;
        c.data2 = 2;
        c.data3 = 3;
        tsta(&c);
        tstb(&c);
        tstc(&c);
        return 0;
    }
    if (0)
    {
        RedirectStdout rs(logpath.c_str());
        Signal::Intervals I(100, 300);
        cout << I.toString() << endl;
        I ^= Signal::Interval(150,150);
        cout << I.toString() << endl;
        I ^= Signal::Interval(50,50);
        cout << I.toString() << endl;
        I ^= Signal::Intervals(50,150);
        cout << I.toString() << endl;
        return 0;
    }

    if (0)
    {
        RedirectStdout rs(logpath.c_str());
        Signal::Intervals I(100, 300);
        cout << I.toString() << endl;
        I -= Signal::Interval(150,150);
        cout << I.toString() << endl;
        I |= Signal::Interval(50,50);
        cout << I.toString() << endl;
        I &= Signal::Intervals(150,150);
        cout << I.toString() << endl;
        return 0;
    }
    if (0)
    {
        RedirectStdout rs(logpath.c_str());
        Intervals I(100, 300);
        vector<Intervals> T;
        T.push_back( Intervals(50,80) );
        T.push_back( Intervals(50,100));
        T.push_back( Intervals(50,200));
        T.push_back( Intervals(50,400));
        T.push_back( Intervals(100,300));
        T.push_back( Intervals(200,250));
        T.push_back( Intervals(200,400));
        T.push_back( Intervals(300,400));
        T.push_back( Intervals(350,400));

        // note operator precendence for bit operators:
        // 'a & b == c' is equivalent to 'a & (b == c)'.
        // Thas i  not '(a & b) == c' which was probably intended.

        BOOST_ASSERT( (I | T[0]) == (Intervals(100,300) | Intervals(50,80)));
        BOOST_ASSERT( (I | T[1]) == Intervals(50,300));
        BOOST_ASSERT( (I | T[2]) == Intervals(50,300));
        BOOST_ASSERT( (I | T[3]) == Intervals(50,400));
        BOOST_ASSERT( (I | T[4]) == Intervals(100,300));
        BOOST_ASSERT( (I | T[5]) == Intervals(100,300));
        BOOST_ASSERT( (I | T[6]) == Intervals(100,400));
        BOOST_ASSERT( (I | T[7]) == Intervals(100,400));
        BOOST_ASSERT( (I | T[8]) == (Intervals(350,400) | Intervals(100,300)));
        BOOST_ASSERT( (I - T[0]) == Intervals(100,300));
        BOOST_ASSERT( (I - T[1]) == Intervals(100,300));
        BOOST_ASSERT( (I - T[2]) == Intervals(200,300));
        BOOST_ASSERT( (I - T[3]) == Intervals());
        BOOST_ASSERT( (I - T[4]) == Intervals());
        BOOST_ASSERT( (I - T[5]) == (Intervals(100,200) | Intervals(250,300)));
        BOOST_ASSERT( (I - T[6]) == Intervals(100,200));
        BOOST_ASSERT( (I - T[7]) == Intervals(100,300));
        BOOST_ASSERT( (I - T[8]) == Intervals(100,300));
        BOOST_ASSERT( (I & T[0]) == Intervals());
        BOOST_ASSERT( (I & T[1]) == Intervals());
        BOOST_ASSERT( (I & T[2]) == Intervals(100,200));
        BOOST_ASSERT( (I & T[3]) == Intervals(100,300));
        BOOST_ASSERT( (I & T[4]) == Intervals(100,300));
        BOOST_ASSERT( (I & T[5]) == Intervals(200,250));
        BOOST_ASSERT( (I & T[6]) == Intervals(200,300));
        BOOST_ASSERT( (I & T[7]) == Intervals());
        BOOST_ASSERT( (I & T[8]) == Intervals());
        TaskInfo("ok");
        return 0;
    }
    if (0)
    {
        std::vector<float> r;
        r.reserve(10);
        TaskInfo("r.size() = %u", r.size() );
        r.push_back(4);
        TaskInfo("r.size() = %u", r.size() );
        return 0;
    }
    if (0)
    {
        {
            TaskTimer tt("Timing tasktimer");
        }
        {
            TaskTimer tt("Timing loop");
            for (unsigned N = 1000; N; --N)
            {
            }
        }
        {
            TaskTimer tt("Timing threadchecker");
            for (unsigned N = 1000; N; --N)
            {
                ThreadChecker tc;
            }
        }
        // Ubuntu, debug build of both gpumisc and sonicawe
        //00:12:20.787 Timing tasktimer... done in 4.0 us.
        //00:12:20.788 Timing loop... done in 6.0 us.
        //00:12:20.788 Timing threadchecker... done in 37.0 us.
        return 0;
    }

    if (0)
    {
        /*const A& a = hej();
        std::cout << "tjo" << std::endl;
        return 0;*/
    }
    if (0) {
        ResampleTest rt;
        rt.test1();
        rt.test2();
        rt.test3();
        rt.test4();
        rt.test5();
        return 0;
    }
	if (0) try {
                /*{
			Signal::pOperation ljud(new Adapters::Audiofile("C:\\dev\\Musik\\music-1.ogg"));

			std::ofstream ofs("tstfil.xml");
			boost::archive::xml_oarchive xml(ofs);
			xml & boost::serialization::make_nvp("hej2", ljud );
		}
		{
			std::ifstream ifs("tstfil.xml");
			boost::archive::xml_iarchive xml(ifs);

			Signal::pOperation ljud;
			xml & boost::serialization::make_nvp("hej2", ljud );
			cout << "filnamn: " << ((Adapters::Audiofile*)ljud.get())->filename() << endl;
                }*/
		return 0;
	} catch (std::exception const& x)
	{
		cout << vartype(x) << ": " << x.what() << endl;
		return 0;
	}



    if(0) {
        TaskTimer tt("Testing supersample");
        Adapters::Audiofile file("testfil.wav");
        pBuffer data = file.read(Interval(0,1));
        Statistics<float> s1(data->waveform_data());

        pBuffer super = Tfr::SuperSample::supersample(data, 8*file.sample_rate());
        tt.info("super %u", super->number_of_samples());
        Statistics<float> s2(super->waveform_data());
        Adapters::WriteWav::writeToDisk("testut.wav", super, false);
        return 0;
    }

    if(0) {
        Gauss g(ResamplePos(-1.1, 20), ResamplePos(1.5, 1.5));
        double s = 0;
        double dx = .1, dy = .1;

        for (double y=10; y<30; y+=dy)
            for (double x=-10; x<10; x+=dx)
                s += g.gauss_value(x, y)*dx*dy;

        printf("1-s=%g\n", (float)(1.f-s));
        return 0;
    }

	

    // Save previous log files
    remove((logdir + "sonicawe~5.log").c_str());
    rename((logdir+"sonicawe~4.log").c_str(), (logdir+"sonicawe~5.log").c_str());
    rename((logdir+"sonicawe~3.log").c_str(), (logdir+"sonicawe~4.log").c_str());
    rename((logdir+"sonicawe~2.log").c_str(), (logdir+"sonicawe~3.log").c_str());
    rename((logdir+"sonicawe~.log").c_str(), (logdir+"sonicawe~2.log").c_str());
    rename(logpath.c_str(), (logdir+"sonicawe~.log").c_str());

    // Write all stdout and stderr to sonicawe.log instead
    boost::shared_ptr<RedirectStdout> rs(new RedirectStdout(logpath.c_str()));

    TaskTimer::setLogLevelStream(TaskTimer::LogVerbose, 0);

    TaskInfo("Starting Sonic AWE");

    QGL::setPreferredPaintEngine(QPaintEngine::OpenGL);

    try {
        Sawe::Application a(argc, argv, true);
        a.rs = rs;
        rs.reset();

        {
            TaskInfo ti("Version: %s", a.version_string().c_str());
            TaskInfo("Build timestamp: %s, %s. Revision %s", __DATE__, __TIME__, SONICAWE_REVISION);

            boost::gregorian::date today = boost::gregorian::day_clock::local_day();
            boost::gregorian::date_facet* facet(new boost::gregorian::date_facet("%A %B %d, %Y"));
            ti.tt().getStream().imbue(std::locale(std::cout.getloc(), facet));
            ti.tt().getStream() << "Program started " << today;
            TaskInfo ti2("%u command line argument%s", argc, argc==1?"":"s");
            for (int i=0; i<argc; ++i)
                TaskInfo("%s", argv[i]);
        }

        // Check if a cuda context can be created, but don't require OpenGL bindings just yet
        if (!check_cuda( false ))
            return -1;

        a.parse_command_line_options(argc, argv);

#ifdef USE_CUDA
        CudaProperties::printInfo(CudaProperties::getCudaDeviceProp());
#endif
        if(0) {
            TaskTimer tt("Cwt inverse");
            Adapters::Audiofile file("chirp.wav");

            Tfr::Cwt& cwt = Tfr::Cwt::Singleton();

            unsigned firstSample = 44100*2;
            unsigned chunk_alignment = cwt.chunk_alignment( file.sample_rate() );
            firstSample = int_div_ceil(firstSample, chunk_alignment)*chunk_alignment;
            unsigned time_support = cwt.wavelet_time_support_samples( file.sample_rate() );

            pBuffer data = file.readFixedLength(Interval(firstSample,firstSample+65536));

            Tfr::pChunk chunk = Tfr::Cwt::Singleton()( data );
            pBuffer inv = cwt.inverse( chunk );

            TaskTimer("%s", inv->getInterval().toString().c_str()).suppressTiming();
            TaskTimer("%s", Interval(
                    firstSample+time_support,
                    firstSample+time_support+inv->number_of_samples()).toString().c_str()).suppressTiming();
            //pBuffer data2 = file.readFixedLength( inv->getInterval() );
            pBuffer data2 = file.readFixedLength(
            Interval(
                    firstSample+time_support,
                    firstSample+time_support+inv->number_of_samples()));

            Statistics<float> s1(data2->waveform_data());
            Statistics<float> si(inv->waveform_data());

            tt.info("firstSample = %u", firstSample);
            tt.info("time_support = %u", time_support);
            Adapters::WriteWav::writeToDisk("invtest.wav", inv, false);
            return 0;
        }

        // Recreate the cuda context and use OpenGL bindings
        if( 0 != QGLContext::currentContext() )
            if (!check_cuda( true ))
                return -1;

        int r = -1;
        if (0 < a.count_projects())
            r = a.exec();

        // When the OpenGL context is destroyed, the Cuda context becomes
        // invalid. Check that some kind of cleanup took place and that the
        // cuda context doesn't think it is still valid.
        // TODO 0 != QGLContext::currentContext() when exiting by an exception
        // that stops the mainloop.
        if( 0 != QGLContext::currentContext() )
            TaskInfo("Error: OpenGL context was not destroyed prior to application exit");

#ifdef USE_CUDA
        CUdevice current_device;
        if( CUDA_ERROR_INVALID_CONTEXT != cuCtxGetDevice( &current_device ))
            TaskInfo("Error: CUDA context was not destroyed prior to application exit");
#endif

        return r;
    } catch (const std::exception &x) {
        if (!rs) rs.reset(new RedirectStdout(logpath.c_str()));
        Sawe::Application::display_fatal_exception(x);
        return -2;
    } catch (...) {
        if (!rs) rs.reset(new RedirectStdout(logpath.c_str()));
        Sawe::Application::display_fatal_exception();
        return -3;
    }
}
