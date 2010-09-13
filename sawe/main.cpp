#include "tfr/cwt.h"
#include <QtGui/QFileDialog>
#include <QtGui/QMessageBox>
#include <iostream>
#include <stdio.h>
#include "saweui/mainwindow.h"
#include "saweui/displaywidget.h"
#include "signal/audiofile.h"
#include "signal/microphonerecorder.h"
#include <CudaProperties.h>
#include <QString>
#include <CudaException.h>
#include "heightmap/renderer.h"
#include "sawe/csv.h"
#include "sawe/hdf5.h"
#include "sawe/application.h"

using namespace std;
using namespace boost;

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
//"    yscale             Tells how to translate the complex transform to a \n"
//"                       hightmap. Valid yscale values:\n"
//"                       0   A=amplitude of CWT coefficients, default\n"
//"                       1   A * exp(.001*fi)\n"
//"                       2   log(1 + |A|)\n"
//"                       3   log(1 + [A * exp(.001*fi)]\n"
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
"                       filters are being used as the GUI won't be lock during\n"
"                       computation.\n"
"\n"
"Sonic AWE, 2010\n";

static unsigned _channel=0;
static unsigned _scales_per_octave = 50;
static float _wavelet_std_t = 0.06f;
static unsigned _samples_per_chunk = 1;
//static float _wavelet_std_t = 0.03;
//static unsigned _samples_per_chunk = (1<<12) - 2*(_wavelet_std_t*44100+31)/32*32-1;
static unsigned _samples_per_block = 1<<7;//                                                                                                    9;
static unsigned _scales_per_block = 1<<8;
static unsigned _yscale = DisplayWidget::Yscale_Linear;
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
        else if (readarg(&cmd, wavelet_std_t));
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


bool check_cuda() {
    stringstream ss;
    void* ptr=(void*)0;
    CudaException namedError(cudaSuccess);
    try {
        CudaException_CALL_CHECK ( cudaMalloc( &ptr, 1024 ));
        CudaException_CALL_CHECK ( cudaFree( ptr ));
        GpuCpuData<float> a( 0, make_cudaExtent(1024,1,1), GpuCpuVoidData::CudaGlobal );
    }
    catch (const CudaException& x) {
        namedError = x;
#ifdef _DEBUG
        ss << x.what() << endl << "Cuda error code " << x.getCudaError() << endl << endl;
#endif
        ptr = 0;
    } catch (...) {
        ss << "catch (...)" << endl;
        ptr = 0;
    }
    
    if (ptr && CudaProperties::haveCuda())
        return true;

    if (cudaErrorInsufficientDriver == namedError.getCudaError())
    {
        ss << "Sonic AWE requires you to have installed newer CUDA-compatible graphics drivers from NVIDIA. "
                << "CUDA drivers are installed on this computer but they are too old. "
                << "You can download new drivers from NVIDIA;" << endl;
    }
    else
    {
        ss   << "Sonic AWE requires you to have installed CUDA-compatible graphics drivers from NVIDIA, and no such driver was found." << endl
             << endl
             << "Hardware requirements: You need to have one of these graphics cards from NVIDIA;" << endl
             << "   www.nvidia.com/object/cuda_gpus.html" << endl
             << endl
             << "Software requirements: You also need to have installed recent display drivers from NVIDIA;" << endl;
    }

    ss
#ifdef __APPLE__
         << "   http://developer.nvidia.com/object/cuda_3_0_downloads.html#MacOS" << endl
#else
         << "   www.nvidia.com" << endl
#endif
         << endl
         << endl
         << "Sonic AWE cannot start." << endl;

    cerr << ss.str();
    cerr.flush();

    QMessageBox::critical( 0,
                 "Couldn't find CUDA, cannot start Sonic AWE",
				 QString::fromLocal8Bit(ss.str().c_str()) );

    return false;
}

void validate_arguments() {
    switch ( _yscale )
    {
        case DisplayWidget::Yscale_Linear:
        case DisplayWidget::Yscale_ExpLinear:
        case DisplayWidget::Yscale_LogLinear:
        case DisplayWidget::Yscale_LogExpLinear:
            break;
        default:
            printf("Invalid yscale value, must be one of {1, 2, 3, 4}\n\n%s", _sawe_usage_string);
            exit(-1);
    }
}


int main(int argc, char *argv[])
{
//    printf("Fastest size = %u\n", Tfr::Stft::build_performance_statistics(true));
//    return 0;

//#ifndef __GNUC__
    TaskTimer::setLogLevelStream(TaskTimer::LogVerbose, 0);
//#endif

    QGL::setPreferredPaintEngine(QPaintEngine::OpenGL);

    Sawe::Application a(argc, argv);
    if (!check_cuda())
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
        cwt.wavelet_std_t( _wavelet_std_t );

        unsigned total_samples_per_chunk = cwt.prev_good_size( 1<<_samples_per_chunk, p->head_source()->sample_rate() );
        TaskTimer("Samples per chunk = %d", total_samples_per_chunk).suppressTiming();

        if (_get_csv != (unsigned)-1) {
            if (0==p->head_source()->number_of_samples()) {
                                Sawe::Application::display_fatal_exception(std::invalid_argument("Can't extract CSV without input file."));
				return -1;
			}

            Sawe::Csv csv;
            csv.source( p->head_source() );
            csv.read( Signal::Interval( _get_csv*total_samples_per_chunk, (_get_csv+1)*total_samples_per_chunk ));
        }

        if (_get_hdf != (unsigned)-1) {
            if (0==p->head_source()->number_of_samples()) {
                            Sawe::Application::display_fatal_exception(std::invalid_argument("Can't extract HDF without input file."));
				return -1;
			}

            Sawe::Hdf5Chunk hdf5;
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

        p->worker->samples_per_chunk_hint( _samples_per_chunk );
        if (_multithread)
            p->worker->start();

        // p->displayWidget()->yscale = (DisplayWidget::Yscale)_yscale;
        // todo p->tools.playback_view.playback_device = _playback_device;
        // todo p->tools.diskwriter_view.selection_filename  = _selectionfile;
        //p->displayWidget()->collection()->samples_per_block( _samples_per_block );
        //p->displayWidget()->collection()->scales_per_block( _scales_per_block );

		a.openadd_project( p );

		p.reset(); // a keeps a copy of pProject

        int r = a.exec();

        // TODO why doesn't this work? CudaException_CALL_CHECK ( cudaThreadExit() );
        return r;
    } catch (const std::exception &x) {
        Sawe::Application::display_fatal_exception(x);
        return -2;
    } catch (...) {
        Sawe::Application::display_fatal_exception();
        return -3;
    }
}

