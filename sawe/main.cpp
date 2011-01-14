// Sonic AWE
#include "sawe/application.h"
#include "tfr/cwt.h"

// gpumisc
#include <CudaProperties.h>
#include <CudaException.h>
#include <redirectstdout.h>

// Qt
#include <QtGui/QMessageBox>
#include <qgl.h>

// cuda
#include <cuda_gl_interop.h>
#include <cuda.h>

using namespace std;
using namespace boost;
using namespace Ui;


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
            TaskInfo("Cuda memory available %g MB (of which %g MB is free to use)",
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
    nvidia_url = "\"Developer Drivers for MacOS\" at " + endl +
                 "http://www.nvidia.com/object/cuda_get.html#MacOS";
#else
    nvidia_url = "www.nvidia.com";
#endif

    stringstream msg;

    switch (namedError.getCudaError())
    {
    case cudaErrorInsufficientDriver:
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
    case cudaErrorDevicesUnavailable:
        msg << "The NVIDIA CUDA driver couldn't start because the GPU is occupied. "
                << "Are you currently using the GPU in any other application? "
                << "If you're not intentionally using the GPU right now the driver might have been left in an inconsistent state after a previous crash. Rebooting your computer could work around this for now. "
                << "Also make sure that you have installed the latest graphics drivers." << endl
                << endl
                << endl
                << "Sonic AWE cannot start. Please try again after closing some other graphics applications.";
        break;
    default:
    {
        cerr << ss.str();
        cerr.flush();

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

    QMessageBox::critical( 0,
                 "Couldn't find CUDA, cannot start Sonic AWE",
                 QString::fromLocal8Bit(msg.str().c_str()) );

    return false;
}



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
    A() { std::cout << __FUNCTION__ << this << std::endl; }
    ~A() { std::cout << __FUNCTION__ << this << std::endl; }
};

A hej()
{
    return A();
}

int main(int argc, char *argv[])
{
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
        Gauss g(make_float2(-1.1, 20), make_float2(1.5, 1.5));
        double s = 0;
        double dx = .1, dy = .1;

        for (double y=10; y<30; y+=dy)
            for (double x=-10; x<10; x+=dx)
                s += g.gauss_value(x, y)*dx*dy;

        printf("1-s=%g\n", (float)(1.f-s));
        return 0;
    }

    // Write all stdout and stderr to sonicawe.log instead
    RedirectStdout rs("sonicawe.log");

    TaskTimer::setLogLevelStream(TaskTimer::LogVerbose, 0);

    TaskInfo("Starting Sonic AWE");

    QGL::setPreferredPaintEngine(QPaintEngine::OpenGL);

    try {
        Sawe::Application a(argc, argv, true);

        {
            TaskInfo ti("Version: %s", a.version_string().c_str());
            TaskInfo("Build timestamp: %s, %s", __DATE__, __TIME__);

            boost::gregorian::date today = boost::gregorian::day_clock::local_day();
            boost::gregorian::date_facet* facet(new boost::gregorian::date_facet("%A %B %d, %Y"));
            ti.tt().getStream().imbue(std::locale(std::cout.getloc(), facet));
            ti.tt().getStream() << "Program started " << today;
        }

        // Check if a cuda context can be created, but don't require OpenGL bindings just yet
        if (!check_cuda( false ))
            return -1;

        a.parse_command_line_options(argc, argv);

        CudaProperties::printInfo(CudaProperties::getCudaDeviceProp());

        if(0) {
            TaskTimer tt("Cwt inverse");
            Adapters::Audiofile file("chirp.wav");

            Tfr::Cwt& cwt = Tfr::Cwt::Singleton();

            unsigned firstSample = 44100*2;
            unsigned c = cwt.find_bin( cwt.nScales( file.sample_rate() ) - 1 );
            firstSample = (firstSample+(1<<c)-1)>>c<<c;
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
        if (!check_cuda( true ))
            return -1;

        int r = a.exec();

        // When the OpenGL context is destroyed, the Cuda context becomes
        // invalid. Check that some kind of cleanup took place and that the
        // cuda context doesn't think it is still valid.
        // TODO 0 != QGLContext::currentContext() when exiting by an exception
        // that stops the mainloop.
        if( 0 != QGLContext::currentContext() )
            TaskInfo("Error: OpenGL context was not destroyed prior to application exit");

        CUdevice current_device;
        if( CUDA_ERROR_INVALID_CONTEXT != cuCtxGetDevice( &current_device ))
            TaskInfo("Error: CUDA context was not destroyed prior to application exit");

        return r;
    } catch (const std::exception &x) {
        Sawe::Application::display_fatal_exception(x);
        return -2;
    } catch (...) {
        Sawe::Application::display_fatal_exception();
        return -3;
    }
}
