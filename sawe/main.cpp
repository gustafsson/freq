#include "sawe/application.h"
#include "tfr/cwt.h"
#include "heightmap/renderer.h"

#include "ui/mainwindow.h"
#include "tools/toolfactory.h"

#include "adapters/audiofile.h"
#include "adapters/microphonerecorder.h"
#include "adapters/csv.h"
#include "adapters/hdf5.h"

#include <CudaProperties.h>
#include <CudaException.h>

#include <iostream>
#include <stdio.h>

#include <QString>
#include <QtGui/QFileDialog>
#include <QtGui/QMessageBox>
#include <QGLContext>
#include <cuda.h>

using namespace std;
using namespace boost;
using namespace Ui;

static const char _sawe_usage_string[] =
        "sonicawe [--parameter=value]* [FILENAME]\n"
        "sonicawe [--parameter] \n"
        "sonicawe [--help] \n"
        "sonicawe [--version] \n"
        "\n"
"    Each parameter takes a value, if no value is given the default value is\n"
"    written to standard output and the program exits immediately after.\n"
"    Valid parameters are:\n"
"\n"
//"    channel            Sonic AWE can only work with one-dimensional data.\n"
//"                       channel specifies which channel to read from if source\n"
//"                       has more than one channel.\n"
"    samples_per_chunk  Only used by extract_chunk option.\n"
"                       The transform is computed in chunks from the input\n"
"                       This determines the number of input samples that \n"
"                       should correspond to one chunk of the transform by\n"
"                       2^samples_per_chunk.\n"
"    scales_per_octave  Accuracy of transform, higher accuracy takes more time\n"
"                       to compute.\n"
"    samples_per_block  The transform chunks are downsampled to blocks for\n"
"                       rendering, this gives the number of samples per block.\n"
"    scales_per_block   Number of scales per block, se samples_per_block.\n"
"    get_csv            Saves the given chunk number into sawe.csv which \n"
"                       then can be read by matlab or octave.\n"
"    get_hdf            Saves the given chunk number into sawe.h5 which \n"
"                       then can be read by matlab or octave.\n"
"    get_chunk_count    Sonic AWE prints number of chunks needed and then exits.\n"
"    wavelet_std_t      Transform chunks used when computing get_* overlap this\n"
"                       much, given in seconds.\n"
"    record             Starts Sonic AWE starts in record mode. [default]\n"
"    record_device      Selects a specific device for recording. -1 specifices\n"
"                       the default input device/microphone.\n"
"    playback_device    Selects a specific device for playback. -1 specifices the\n"
"                       default output device.\n"
"    multithread        If set, starts a parallell worker thread. Good if heavy \n"
"                       filters are being used as the GUI won't be locked during\n"
"                       computation.\n"
"\n"
"Sonic AWE, 2010\n";

static unsigned _channel=0;
static unsigned _scales_per_octave = 20;
static float _wavelet_time_support = 3;
static unsigned _samples_per_chunk = 1;
static unsigned _samples_per_block = 1<<7;//                                                                                                    9;
static unsigned _scales_per_block = 1<<8;
static unsigned _get_hdf = (unsigned)-1;
static unsigned _get_csv = (unsigned)-1;
static bool _get_chunk_count = false;
static std::string _selectionfile = "selection.wav";
static bool _record = false;
static int _record_device = -1;
static int _playback_device = -1;
static std::string _soundfile = "";
static bool _multithread = false;
static bool _sawe_exit=false;

static int prefixcmp(const char *a, const char *prefix) {
    for(;*a && *prefix;a++,prefix++) {
        if (*a < *prefix) return -1;
        if (*a > *prefix) return 1;
    }
    return 0!=*prefix;
}


void atoval(const char *cmd, bool& val) {
    val = (0!=atoi(cmd));
}
void atoval(const char *cmd, float& val) {
    val = atof(cmd);
}
void atoval(const char *cmd, unsigned& val) {
    val = atoi(cmd);
}
void atoval(const char *cmd, int& val) {
    val = atoi(cmd);
}

#define readarg(cmd, name) tryreadarg(cmd, "--"#name, #name, _##name)

template<typename Type>
bool tryreadarg(const char **cmd, const char* prefix, const char* name, Type &val) {
    if (prefixcmp(*cmd, prefix))
        return 0;
    *cmd += strlen(prefix);
    if (**cmd == '=')
        atoval(*cmd+1, val);
    else {
        cout << "default " << name << "=" << val << endl;
        _sawe_exit = true;
    }
    return 1;
}

template<>
bool tryreadarg(const char **cmd, const char* prefix, const char*, bool &val) {
    if (prefixcmp(*cmd, prefix))
        return 0;
    *cmd += strlen(prefix);
    val = true;
    return 1;
}

static int handle_options(char ***argv, int *argc)
{
    int handled = 0;

    while (*argc > 0) {
        const char *cmd = (*argv)[0];
        if (cmd[0] != '-')
            break;

        if (!strcmp(cmd, "--help")) {
            printf("%s", _sawe_usage_string);
            _sawe_exit = true;
        } else if (!strcmp(cmd, "--version")) {
            printf("%s\n", Sawe::Application::version_string().c_str());
            _sawe_exit = true;
        }
        else if (readarg(&cmd, samples_per_chunk));
        else if (readarg(&cmd, scales_per_octave));
        else if (readarg(&cmd, wavelet_time_support));
        else if (readarg(&cmd, samples_per_block));
        else if (readarg(&cmd, scales_per_block));
        // else if (readarg(&cmd, yscale)); // TODO remove?
        else if (readarg(&cmd, get_chunk_count));
        else if (readarg(&cmd, record_device));
        else if (readarg(&cmd, record));
        else if (readarg(&cmd, playback_device));
        else if (readarg(&cmd, channel));
        else if (readarg(&cmd, get_hdf));
        else if (readarg(&cmd, get_csv));
        else if (readarg(&cmd, multithread));
        // TODO use _selectionfile
        else {
            fprintf(stderr, "Unknown option: %s\n", cmd);
            printf("%s", _sawe_usage_string);
            exit(1);
        }

        (*argv)++;
        (*argc)--;
        handled++;
    }

    if (_sawe_exit)
        exit(0);

    return handled;
}


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
    nvidia_url = "http://www.nvidia.com/object/cuda_get.html#MacOS";
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


void validate_arguments()
{
    // TODO
    return;
}

#include "heightmap/resampletest.h"
#include "tools/support/brushpaint.cu.h"
#include "tfr/supersample.h"
#include <Statistics.h>
#include "adapters/audiofile.h"
#include "adapters/writewav.h"
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <fstream>
#include <boost/algorithm/string.hpp>

using namespace Signal;

int main(int argc, char *argv[])
{
	if (0) try {
		{
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
		}
		return 0;
	} catch (std::exception const& x)
	{
		cout << vartype(x) << ": " << x.what() << endl;
		return 0;
	}

    if(0) {
        TaskTimer tt("Cwt inverse");
        Adapters::Audiofile file("chirp.wav");

        Tfr::Cwt& cwt = Tfr::Cwt::Singleton();
        cwt.scales_per_octave( _scales_per_octave );
        cwt.wavelet_time_support( _wavelet_time_support );

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

    //#ifndef __GNUC__
    TaskTimer::setLogLevelStream(TaskTimer::LogVerbose, 0);
//#endif

    QGL::setPreferredPaintEngine(QPaintEngine::OpenGL);

    Sawe::Application a(argc, argv);

    TaskTimer("Starting %s", a.version_string().c_str()).suppressTiming();

    if (!check_cuda( false ))
        return -1;

    // skip application filename
    argv++;
    argc--;

    while (argc) {
        handle_options(&argv, &argc);

        if (argc) {
            if (_soundfile.empty()) {
                _soundfile = argv[0];
            } else {
                fprintf(stderr, "Unknown option: %s\n", argv[0]);
                fprintf(stderr, "Sonic AWE takes only one file (%s) as input argument.\n", _soundfile.c_str());
                printf("%s", _sawe_usage_string);
                exit(1);
            }
            argv++;
            argc--;
        }
    }

    validate_arguments();

    try {
        CudaProperties::printInfo(CudaProperties::getCudaDeviceProp());

        {
            ResampleTest resampletest;

            //resampletest.test4();

            //return 0;
        }

        { // TODO remove?
            TaskTimer tt("Building performance statistics for %s", CudaProperties::getCudaDeviceProp().name);
            tt.info("Fastest size = %u", Tfr::Stft::build_performance_statistics(true, 2));
        }

        Sawe::pProject p; // p goes out of scope before a.exec()

        if (!_soundfile.empty())
			p = Sawe::Project::open( _soundfile );

        if (!p)
            p = Sawe::Project::createRecording( _record_device );

        if (!p)
            return -1;

        // TODO use _channel

        Tfr::Cwt& cwt = Tfr::Cwt::Singleton();
        cwt.scales_per_octave( _scales_per_octave );
        cwt.wavelet_time_support( _wavelet_time_support );

        unsigned total_samples_per_chunk = cwt.prev_good_size( 1<<_samples_per_chunk, p->head_source()->sample_rate() );
        TaskTimer("Samples per chunk = %d", total_samples_per_chunk).suppressTiming();

        if (_get_csv != (unsigned)-1) {
            if (0==p->head_source()->number_of_samples()) {
                                Sawe::Application::display_fatal_exception(std::invalid_argument("Can't extract CSV without input file."));
				return -1;
			}

            Adapters::Csv csv;
            csv.source( p->head_source() );
            csv.read( Signal::Interval( _get_csv*total_samples_per_chunk, (_get_csv+1)*total_samples_per_chunk ));
        }

        if (_get_hdf != (unsigned)-1) {
            if (0==p->head_source()->number_of_samples()) {
                            Sawe::Application::display_fatal_exception(std::invalid_argument("Can't extract HDF without input file."));
				return -1;
			}

            Adapters::Hdf5Chunk hdf5;
            hdf5.source( p->head_source() );
            hdf5.read( Signal::Interval(_get_hdf*total_samples_per_chunk, (_get_hdf+1)*total_samples_per_chunk ));
        }

        if (_get_chunk_count != false) {
            cout << p->head_source()->number_of_samples() / total_samples_per_chunk << endl;
        }

        if (_get_hdf != (unsigned)-1 ||
            _get_csv != (unsigned)-1 ||
            _get_chunk_count != false)
        {
            return 0;
        }

        //p->worker.samples_per_chunk_hint( _samples_per_chunk );
        if (_multithread)
            p->worker.start();

		a.openadd_project( p );

        p->mainWindow(); // Ensures that an OpenGL context is created

        BOOST_ASSERT( QGLContext::currentContext() );

        // Recreate the cuda context and use OpenGL bindings
        if (!check_cuda( true ))
            return -1;

        Tools::ToolFactory &tools = p->tools();

        tools.playback_model.playback_device = _playback_device;
        tools.playback_model.selection_filename  = _selectionfile;
        tools.render_model.collection->samples_per_block( _samples_per_block );
        tools.render_model.collection->scales_per_block( _scales_per_block );

        p.reset(); // 'a' keeps a copy of pProject

        int r = a.exec();

        // When the OpenGL context is destroyed, the Cuda context becomes
        // invalid. Check that some kind of cleanup took place and that the
        // cuda context doesn't think it is still valid.
        // TODO 0 != QGLContext::currentContext() when exiting by an exception
        // that stops the mainloop.
        if( 0 != QGLContext::currentContext() )
			TaskTimer("Error: OpenGL context was not detroyed prior to application exit").suppressTiming();

		if( CUDA_ERROR_INVALID_CONTEXT != cuCtxGetDevice( 0 ))
			TaskTimer("Error: CUDA context was not detroyed prior to application exit").suppressTiming();

        return r;
    } catch (const std::exception &x) {
        Sawe::Application::display_fatal_exception(x);
        return -2;
    } catch (...) {
        Sawe::Application::display_fatal_exception();
        return -3;
    }
}
