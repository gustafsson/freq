#include <QtGui/QApplication>
#include "tfr-cwt.h"
#include <QtGui/QFileDialog>
#include <QtGui/QMessageBox>
#include <QTime>
#include <iostream>
#include <stdio.h>
#include "mainwindow.h"
#include "displaywidget.h"
#include "signal-audiofile.h"
#include "signal-microphonerecorder.h"
#include <sstream>
#include <CudaProperties.h>
#include <QtGui/QMessageBox>
#include <QString>
#include <CudaException.h>
#include "heightmap-renderer.h"
#include "sawe-csv.h"
#include "signal-audiofile.h"
#include "signal-microphonerecorder.h"
#include <demangle.h>

using namespace std;
using namespace boost;

static string _sawe_version_string(
        "Sonic AWE - development snapshot\n");

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
"    wavelet_std_t      Transform chunks overlap this much, given in seconds.\n"
"    samples_per_block  The transform chunks are downsampled to blocks for\n"
"                       rendering, this gives the number of samples per block.\n"
"    scales_per_block   Number of scales per block, se samples_per_block.\n"
"    yscale             Tells how to translate the complex transform to a \n"
"                       hightmap. Valid yscale values:\n"
"                       0   A=amplitude of CWT coefficients, default\n"
"                       1   A * exp(.001*fi)\n"
"                       2   log(1 + |A|)\n"
"                       3   log(1 + [A * exp(.001*fi)]\n"
"    extract_chunk      Saves the given chunk number into sonicawe-n.csv which \n"
"                       then can be read by matlab or octave.\n"
"    get_chunk_count    If assigned a value, Sonic AWE exits with the number of \n"
"                       chunks as exit code.\n"
"    record             If assigned a non-negative value, Sonic AWE records from the \n"
"                       given input device, a value of -1 specifies the default \n"
"                       microphone.\n"
"    playback           Selects a specific device for playback. -1 specifices the\n"
"                       default output device.\n"
"\n"
"Sonic AWE, 2010\n";

static unsigned _channel=0;
static unsigned _scales_per_octave = 40;
//static float _wavelet_std_t = 0.1;
static float _wavelet_std_t = 0.03;
static unsigned _samples_per_chunk = 13;
//static float _wavelet_std_t = 0.03;
//static unsigned _samples_per_chunk = (1<<12) - 2*(_wavelet_std_t*44100+31)/32*32-1;
static unsigned _samples_per_block = 1<<7;//                                                                                                    9;
static unsigned _scales_per_block = 1<<8;
static unsigned _yscale = DisplayWidget::Yscale_Linear;
static unsigned _extract_chunk = (unsigned)-1;
static unsigned _get_chunk_count = (unsigned)-1;
static std::string _selectionfile = "selection.wav";
static int _record = -2;
static int _playback = -1;
static std::string _soundfile = "";
static bool _sawe_exit=false;
std::string fatal_error;

static int prefixcmp(const char *a, const char *prefix) {
    for(;*a && *prefix;a++,prefix++) {
        if (*a < *prefix) return -1;
        if (*a > *prefix) return 1;
    }
    return 0!=*prefix;
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
            printf("%s\n", _sawe_version_string.c_str());
            _sawe_exit = true;
        }
        else if (readarg(&cmd, samples_per_chunk));
        else if (readarg(&cmd, scales_per_octave));
        else if (readarg(&cmd, wavelet_std_t));
        else if (readarg(&cmd, samples_per_block));
        else if (readarg(&cmd, scales_per_block));
        else if (readarg(&cmd, yscale));
        else if (readarg(&cmd, extract_chunk));
        else if (readarg(&cmd, get_chunk_count));
        else if (readarg(&cmd, record));
        else if (readarg(&cmd, playback));
        else if (readarg(&cmd, channel));
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

#define STRINGIFY(x) #x
#define TOSTR(x) STRINGIFY(x)

void fatal_exception_cerr( const std::string& str )
{
    cerr << endl << endl
         << "======================" << endl
         << str << endl
         << "======================" << endl;
    cerr.flush();
}

void fatal_exception_qt( const std::string& str )
{
    QMessageBox::critical( 0,
                 QString("Fatal error. Sonic AWE needs to close"),
                 QString::fromStdString(str) );
}

void fatal_exception( const std::string& str )
{
    fatal_exception_cerr(str);
    fatal_exception_qt(str);
}

string fatal_exception( const std::exception &x )
{
    std::stringstream ss;
    ss   << "Error: " << demangle(typeid(x).name()) << endl
         << "Message: " << x.what();
    return ss.str();
}

string fatal_unknown_exception() {
    return "Error: An unknown error occurred";
}


class SonicAWE_Application: public QApplication
{
public:
    SonicAWE_Application( int& argc, char **argv)
    :   QApplication(argc, argv)
    {
    }

    virtual bool notify(QObject * receiver, QEvent * e) {
        bool v = false;
        try {
            if(!fatal_error.empty())
                this->exit(-2);

            v = QApplication::notify(receiver,e);
        } catch (const std::exception &x) {
            if(fatal_error.empty())
                fatal_exception_cerr( fatal_error = fatal_exception(x) );
            this->exit(-2);
        } catch (...) {
            if(fatal_error.empty())
                fatal_exception_cerr( fatal_error = fatal_unknown_exception() );
            this->exit(-2);
        }
        return v;
    }
};

bool check_cuda() {
    stringstream ss;
    void* ptr=(void*)0;
    try {
        CudaException_CALL_CHECK ( cudaMalloc( &ptr, 1024 ));
        CudaException_CALL_CHECK ( cudaFree( ptr ));
        GpuCpuData<float> a( 0, make_cudaExtent(1024,1,1), GpuCpuVoidData::CudaGlobal );
    }
    catch (const CudaException& x) {
#ifdef _DEBUG
        ss << x.what() << endl;
#endif
        ptr = 0;
    } catch (...) {
        ss << "ptr=0" << endl;
        ptr = 0;
    }
    
    if (ptr && CudaProperties::haveCuda())
        return true;

    ss   << "Sonic AWE requires you to have installed CUDA-compatible graphics drivers from NVIDIA, and no such driver was found." << endl
         << endl
         << "Hardware requirements: You need to have one of these graphics cards from NVIDIA;" << endl
         << "   www.nvidia.com/object/cuda_gpus.html" << endl
         << endl
         << "Software requirements: You also need to have installed recent display drivers from NVIDIA;" << endl
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
                 QString::fromStdString(ss.str()) );

    return false;
}

void validate_arguments() {
    if (-1>_record) if (0 == _soundfile.length() || !QFile::exists(_soundfile.c_str())) {
        QString fileName = QFileDialog::getOpenFileName(0, "Open sound file", NULL, QString(Signal::getFileFormatsQtFilter().c_str()));
        if (0 == fileName.length())
            exit(0);
        _soundfile = fileName.toStdString();
    }

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
//#ifndef __GNUC__
    TaskTimer::setLogLevelStream(TaskTimer::LogVerbose, 0);
//#endif

    QDateTime now = QDateTime::currentDateTime();
    now.date().year();
    stringstream ss;
    ss << "Sonic AWE";
#ifndef SONICAWE_RELEASE
    ss << " - ";
#ifdef SONICAWE_VERSION
    ss << TOSTR(SONICAWE_VERSION);
#else
    ss << __DATE__;// << " - " << __TIME__;
#endif
#endif

#ifdef SONICAWE_BRANCH
    if( 0 < strlen( TOSTR(SONICAWE_BRANCH) ))
        ss << " - branch: " << TOSTR(SONICAWE_BRANCH);
#endif

    _sawe_version_string = ss.str();

    SonicAWE_Application a(argc, argv);
    if (!check_cuda())
        return -1;

    MainWindow w(_sawe_version_string.c_str());
    
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

        boost::shared_ptr<Signal::Source> wf;

        if (-1<=_record)
            // TODO use _channel
            wf.reset( new Signal::MicrophoneRecorder(_record) );
        else {
            // TODO use _channel
            printf("Reading file: %s\n", _soundfile.c_str());
            wf.reset( new Signal::Audiofile( _soundfile.c_str() ) );
        }

        unsigned redundant = 2*(((unsigned)(_wavelet_std_t*wf->sample_rate())+31)/32*32);
        while ( (unsigned)(1<<_samples_per_chunk) < redundant ) {
            _samples_per_chunk++;
            TaskTimer("To few samples per chunk, increasing to 2^%d", _samples_per_chunk).suppressTiming();
        }
        unsigned total_samples_per_chunk = (1<<_samples_per_chunk) - redundant;

        Tfr::pCwt cwt = Tfr::CwtSingleton::instance();
        cwt->scales_per_octave( _scales_per_octave );
        cwt->wavelet_std_t( _wavelet_std_t );

        if (_extract_chunk != (unsigned)-1) {
            Sawe::Csv().put( wf->read( _extract_chunk*total_samples_per_chunk - redundant/2, (1<<_samples_per_chunk) ));
            return 0;
        }

        if (_get_chunk_count != (unsigned)-1) {
            return wf->number_of_samples() / total_samples_per_chunk;
        }

        Signal::pWorker wk( new Signal::Worker( wf ) );
        Heightmap::Collection* sgp( new Heightmap::Collection(wk) );
        Signal::pSink sg( sgp );
        sgp->samples_per_block( _samples_per_block );
        sgp->scales_per_block( _scales_per_block );
        boost::shared_ptr<DisplayWidget> dw( new DisplayWidget( wk, sg, _playback, _selectionfile, 0 ) );
        dw->yscale = (DisplayWidget::Yscale)_yscale;

        w.connectLayerWindow(dw.get());
        w.setCentralWidget( dw.get() );
        dw->show();
        w.show();

        int r = a.exec();
        if (!fatal_error.empty())
            fatal_exception_qt(fatal_error);

        // TODO why doesn't this work? CudaException_CALL_CHECK ( cudaThreadExit() );
        return r;
    } catch (const std::exception &x) {
        fatal_exception(fatal_exception(x));
        return -2;
    } catch (...) {
        fatal_exception(fatal_unknown_exception());
        return -3;
    }
}

