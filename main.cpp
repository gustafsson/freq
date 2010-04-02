#include <QtGui/QApplication>
#include "transform.h"
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
#include <CudaException.h>
//#include <cuda_runtime.h>

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
"    samples_per_chunk  The transform is computed in chunks from the input\n"
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
"    record             If assigned a non-zero value, Sonic AWE record from the \n"
"                       default microphone.\n"
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
static unsigned _record = 0;
static std::string _soundfile = "";
static std::string _playback_source_test = "";
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

void fatal_exception( const std::string& str )
{
    cerr << endl << endl
         << "======================" << endl
         << str
         << "======================" << endl;
    cerr.flush();

    QMessageBox::critical( 0,
                 QString("Fatal error. Sonic AWE needs to close"),
                 QString::fromStdString(str) );
}
void fatal_exception( const std::exception &x )
{
    stringstream ss;
    ss   << "Error: " << typeid(x).name() << endl
         << "Message: " << x.what() << endl;

    fatal_exception( ss.str() );
}

void fatal_unknown_exception() {
    fatal_exception( string("An unknown error occurred") );
}

class SonicAWE_Application: public QApplication
{
public:
    SonicAWE_Application( int& argc, char **argv)
    :   QApplication(argc, argv)
    {}

    virtual bool notify(QObject * receiver, QEvent * e) {
        bool v = false;
        try {
            v = QApplication::notify(receiver,e);
        } catch (const std::exception &x) {
            fatal_error = "Error: ";
            fatal_error.append(typeid(x).name());
            fatal_error.append("\n");
            fatal_error.append("Message: ");
            fatal_error.append(x.what());
            fatal_error.append("\n");
            this->exit(-2);
        } catch (...) {
            fatal_error = "An unknown error occurred";
            this->exit(-2);
        }
        return v;
    }
};

bool check_cuda() {
    stringstream ss;
    void* ptr=(void*)1;
    try {
        CudaException_CALL_CHECK ( cudaMalloc( &ptr, 1024 ));
        CudaException_CALL_CHECK ( cudaFree( ptr ));
        GpuCpuData<float> a( 0, make_cudaExtent(1024,1,1), GpuCpuVoidData::CudaGlobal );
    }
    catch (const CudaException& x) {
        ss << x.what() << endl;
        ptr = 0;
    } catch (...) {
        ss << "ptr=0" << endl;
        ptr = 0;
    }
    
    if (ptr && CudaProperties::haveCuda())
        return true;

    ss   << "Sonic AWE requires you to have installed graphics drivers from NVIDIA." << endl
         << endl
         << "You need to have one of these graphics cards from NVIDIA;" << endl
         << "   www.nvidia.com/object/cuda_gpus.html" << endl
         << endl
         << "You also need to have NVIDIAs display drivers installed;" << endl
         <<"    www.nvidia.com" << endl
         << endl
         << endl
         << "Sonic AWE cannot start." << endl;
    cerr << ss.str();
    cerr.flush();

    QMessageBox::critical( 0,
                 "Cannot start Sonic AWE",
                 QString::fromStdString(ss.str()) );

    return false;
}

void validate_arguments() {
    if (!_record) if (0 == _soundfile.length() || !QFile::exists(_soundfile.c_str())) {
        QString fileName = QFileDialog::getOpenFileName(0, "Open sound file");
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
    TaskTimer::setLogLevelStream(TaskTimer::LogVerbose, 0);
  
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
                _playback_source_test =  argv[0];;
            }
            argv++;
            argc--;
        }
    }

    validate_arguments();

    try {
        boost::shared_ptr<Signal::Source> wf;

        if (_record)
            wf.reset( new Signal::MicrophoneRecorder() );
        else {
            printf("Reading file: %s\n", _soundfile.c_str());
            wf.reset( new Signal::Audiofile( _soundfile.c_str() ) );
        }

        // TODO compute required memory by application and adjust _samples_per_chunk thereafter
        //unsigned mem = CudaProperties::getCudaDeviceProp( CudaProperties::getCudaCurrentDevice() ).totalGlobalMem;

        unsigned redundant = 2*(((unsigned)(_wavelet_std_t*wf->sample_rate())+31)/32*32);
        while ( (unsigned)(1<<_samples_per_chunk) < redundant ) {
            _samples_per_chunk++;
            TaskTimer("To few samples per chunk, increasing to 2^%d", _samples_per_chunk).suppressTiming();
        }
        unsigned total_samples_per_chunk = (1<<_samples_per_chunk) - redundant;

        boost::shared_ptr<Transform> wt( new Transform(wf, _channel, total_samples_per_chunk, _scales_per_octave, _wavelet_std_t ) );

        if (_extract_chunk != (unsigned)-1) {
            wt->saveCsv(_extract_chunk);
            return 0;
        }

        if (_get_chunk_count != (unsigned)-1) {
            return wt->getChunkIndex( wf->number_of_samples() );
        }

        boost::shared_ptr<Spectrogram> sg( new Spectrogram(wt, _samples_per_block, _scales_per_block  ) );
        boost::shared_ptr<DisplayWidget> dw( new DisplayWidget( sg, 0, _playback_source_test ) );
        dw->yscale = (DisplayWidget::Yscale)_yscale;

        w.connectLayerWindow(dw.get());
        w.setCentralWidget( dw.get() );
        dw->show();
        w.show();

        int r = a.exec();
        if (!fatal_error.empty())
            fatal_exception(fatal_error);
        return r;
    } catch (const std::exception &x) {
        fatal_exception(x);
        return -2;
    } catch (...) {
        fatal_unknown_exception();
        return -3;
    }
}

