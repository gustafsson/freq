// Sonic AWE
#include "sawe/application.h"
#include "tfr/cwt.h"
#include "sawe/reader.h"
#include "sawe/configuration.h"
#include "test/unittest.h"

// gpumisc
#include <redirectstdout.h>
#include <neat_math.h>
#include <computationkernel.h>
#include <ThreadChecker.h>
#include "prettifysegfault.h"

// Qt
#include <QtGui/QMessageBox>
#include <qgl.h>
#include <QDesktopServices>
#include <QDir>
#include <QHostInfo>

// boost
#include <boost/date_time/posix_time/posix_time.hpp>

using namespace std;
using namespace boost;
using namespace Ui;
using namespace Signal;


#ifndef USE_CUDA
    static bool check_cuda( bool /*use_OpenGL_bindings*/ ) {
        return true;
    }
#else

// gpumisc
#include "CudaProperties.h"
#include "CudaException.h"
#include "cudaglobalstorage.h"

// cuda
#include <cuda_gl_interop.h>
#include <cuda.h>

static bool check_cuda( bool use_OpenGL_bindings ) {
    stringstream ss;
    void* ptr=(void*)0;
    cudaError namedError = cudaSuccess;

    int runtimeVersion = CUDART_VERSION;
    int driverVersion = -2;

    try {
        CudaException_SAFE_CALL( cudaDriverGetVersion (&driverVersion) );

        if (driverVersion < runtimeVersion)
            throw CudaException(cudaErrorInsufficientDriver);

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

            DataStorage<float> a( 1024 );
            CudaGlobalStorage::ReadWrite<float,1>(&a).getCudaPitchedPtr();

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
        namedError = x.getCudaError();

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
    switch (namedError)
    {
    case cudaErrorMemoryAllocation:
        title << "Out of graphics memory";
        msg << "Cuda error: " << cudaGetErrorString(namedError) << endl
                << endl
                << "If you're currently running other graphics intensive applications or computational software powered by CUDA it might help to close them. " << endl
                << endl
                << "Sonic AWE cannot start. Restart your computer and try again.";
        break;
    case cudaErrorInsufficientDriver:
        title << "Display drivers to old";
        msg << "Cuda error: " << cudaGetErrorString(namedError) << endl
                << endl
                << "Sonic AWE requires you to have installed more recent display drivers from NVIDIA. "
                << "Display drivers from NVIDIA are installed on this computer but they are too old. "
                << "Please download new drivers from NVIDIA:" << endl
                << endl
                << nvidia_url << endl
                << endl;
        if (driverVersion > 0)
        {
            msg
                << "Found cuda driver: v" << (driverVersion/1000) << "." << (driverVersion%1000)/10;
            if (driverVersion%10)
                msg << "." << driverVersion%10;
            msg << endl;
        }
        msg
                << "Minimum required cuda driver: v" << (runtimeVersion/1000) << "." << (runtimeVersion%1000)/10;
        if (runtimeVersion%10)
           msg << "." << runtimeVersion%10;
        msg
                << endl
                << endl
                << "Sonic AWE cannot start. Please try again with updated drivers.";
        break;
#if 3000 < CUDART_VERSION
    case cudaErrorDevicesUnavailable:
        title << "Graphics adapter (GPU) occupied";
        msg << "The NVIDIA CUDA driver could not start because the GPU is occupied. "
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

        title << "Could not find CUDA, cannot start Sonic AWE";
        msg     << "Your computer can't run GPGPU accelerated Sonic AWE" << endl
                << endl
                << "Hardware requirements: You need to have one of these graphics cards from NVIDIA:" << endl
                << "   www.nvidia.com/object/cuda_gpus.html" << endl
                << endl
                << "Software requirements: You also need to have installed recent display drivers from NVIDIA:" << endl
                << endl
                << nvidia_url << endl
                << endl
                << "Sonic AWE cannot start. Ensure you've fulfilled the hardware requirements and try again with updated drivers.";
    }
    }

    TaskInfo("%s\n%s", title.str().c_str(), msg.str().c_str());
    QMessageBox::critical( 0,
                 QString::fromLocal8Bit(title.str().c_str()),
                 QString::fromLocal8Bit(msg.str().c_str()) );

    return false;
}
#endif // USE_CUDA

#ifdef USE_CUDA
#include "heightmap/resampletest.h"
#endif

#include "tools/support/brushpaintkernel.h" // test class Gauss
#include "tfr/supersample.h"
#include <Statistics.h>
#include "adapters/audiofile.h"
#include "adapters/writewav.h"
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <omp.h>

using namespace Signal;


int main(int argc, char *argv[])
{
#ifdef _DEBUG
    omp_set_num_threads(1);
#endif

    PrettifySegfault::setup ();

    if (argc == 2 && 0 == strcmp(argv[1],"--test"))
        return Test::UnitTest::test ();

#ifdef USE_CUDA
    if (0) {
        ResampleTest rt;
        rt.test1();
        rt.test2();
        rt.test3();
        rt.test4();
        rt.test5();
        return 0;
    }
#endif
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


    QGL::setPreferredPaintEngine(QPaintEngine::OpenGL);

    boost::shared_ptr<RedirectStdout> rs;
    std::string logpath;

    try {
        Sawe::Application a(argc, argv, true);

        QString localAppDir = Sawe::Application::log_directory();
        if (QDir(localAppDir).exists()==false)
            QDir().mkpath(localAppDir);

        std::string logdir = (localAppDir + QDir::separator()).toLatin1().data();
        logpath = logdir + "sonicawe.log";
    #ifndef _MSC_VER
        //The following line hinders the redirection from working in windows
        cout << "Saving log file at \"" << logpath << "\"" << endl;
    #endif

        // Save previous log files
        remove((logdir+"sonicawe~5.log").c_str());
        rename((logdir+"sonicawe~4.log").c_str(), (logdir+"sonicawe~5.log").c_str());
        rename((logdir+"sonicawe~3.log").c_str(), (logdir+"sonicawe~4.log").c_str());
        rename((logdir+"sonicawe~2.log").c_str(), (logdir+"sonicawe~3.log").c_str());
        rename((logdir+"sonicawe~.log").c_str(), (logdir+"sonicawe~2.log").c_str());
        rename(logpath.c_str(), (logdir+"sonicawe~.log").c_str());

        // Write all stdout and stderr to sonicawe.log instead
        rs.reset(new RedirectStdout(logpath.c_str()));

        TaskTimer::setLogLevelStream(TaskTimer::LogVerbose, 0);

        TaskInfo("Starting Sonic AWE");


        a.rs = rs;
        rs.reset();

        a.logSystemInfo(argc, argv);

        // Check if a cuda context can be created, but don't require OpenGL bindings just yet
        if (!check_cuda( false ))
            return -17;

        TaskInfo("computation device: %s", Sawe::Configuration::computationDeviceName().c_str());

        if( 0 == a.shared_glwidget()->context())
        {
            QMessageBox::critical(0,
                                  "Sorry, Sonic AWE could not start",
                                  "Failed to initialize OpenGL. You could try updating the drivers for your graphics card."
                                  );
            return -1;
        }

        a.execute_command_line_options();

#ifdef USE_CUDA
        CudaProperties::printInfo(CudaProperties::getCudaDeviceProp());
#endif
        if(0) {
            TaskTimer tt("Cwt inverse");
            Adapters::Audiofile file("chirp.wav");

            Tfr::Cwt cwt;

            unsigned firstSample = 44100*2;
            unsigned chunk_alignment = cwt.chunk_alignment( file.sample_rate() );
            firstSample = int_div_ceil(firstSample, chunk_alignment)*chunk_alignment;
            unsigned time_support = cwt.wavelet_time_support_samples( file.sample_rate() );

            pMonoBuffer data = file.readFixedLength(Interval(firstSample,firstSample+65536))->getChannel (0);

            Tfr::pChunk chunk = cwt( data);
            pMonoBuffer inv = cwt.inverse( chunk );

            TaskTimer("%s", inv->getInterval().toString().c_str()).suppressTiming();
            TaskTimer("%s", Interval(
                    firstSample+time_support,
                    firstSample+time_support+inv->number_of_samples()).toString().c_str()).suppressTiming();
            //pBuffer data2 = file.readFixedLength( inv->getInterval() );
            pMonoBuffer data2 = file.readFixedLength(
            Interval(
                    firstSample+time_support,
                    firstSample+time_support+inv->number_of_samples()))->getChannel (0);

            Statistics<float> s1(data2->waveform_data());
            Statistics<float> si(inv->waveform_data());

            tt.info("firstSample = %u", firstSample);
            tt.info("time_support = %u", time_support);
            Adapters::WriteWav::writeToDisk("invtest.wav", pBuffer(new Signal::Buffer(inv)), false);
            return 0;
        }

        int r = 0;
        if (a.has_other_projects_than(0))
        {
            if( 0 == QGLContext::currentContext())
            {
                QMessageBox::critical(0,
                                      "Sorry, Sonic AWE could not start",
                                      "OpenGL was only partly initialized. You could try updating the drivers for your graphics card."
                                      );
                return -1;
            }

            // Recreate the cuda context and use OpenGL bindings
            if (!check_cuda( true ))
                // check_cuda displays error messages
                return -1;

            r = a.exec();
        }

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
        if (!rs && !logpath.empty()) rs.reset(new RedirectStdout(logpath.c_str()));
        Sawe::Application::display_fatal_exception(x);
        return -2;
    } catch (...) {
        if (!rs && !logpath.empty()) rs.reset(new RedirectStdout(logpath.c_str()));
        Sawe::Application::display_fatal_exception();
        return -3;
    }
}
