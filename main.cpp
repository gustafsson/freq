#include <QtGui/QApplication>
#include "transform.h"
#include <QtGui/QFileDialog>
#include <QTime>
#include <iostream>
#include <stdio.h>
#include "mainwindow.h"
#include "displaywidget.h"
#include <sstream>
#include <CudaProperties.h>
#include <QtGui/QMessageBox>

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
"                       should correspond to one chunk of the transform.\n"
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
"\n"
"Sonic AWE, 2010\n";

static unsigned _channel=0;
static unsigned _scales_per_octave = 40;
static float _wavelet_std_t = 0.06f;
static unsigned _samples_per_chunk = (1<<14) - 2*(((unsigned)(_wavelet_std_t*44100)+31)/32*32);
//static float _wavelet_std_t = 0.03f;
//static unsigned _samples_per_chunk = (1<<13) - 2*(((unsigned)(_wavelet_std_t*44100)+31)/32*32);
//static float _wavelet_std_t = 0.f;
//static unsigned _samples_per_chunk = 1<<15;
static unsigned _samples_per_block = 1<<7;//                                                                                                    9;
static unsigned _scales_per_block = 1<<8;
static unsigned _yscale = DisplayWidget::Yscale_Linear;
static unsigned _extract_chunk = (unsigned)-1;
static unsigned _get_chunk_count = (unsigned)-1;
static std::string _soundfile = "";
static std::string _playback_source_test = "";
static bool _sawe_exit=false;

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

    ::exit(-2);
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
            QApplication::notify(receiver,e);
        } catch (const std::exception &x) {
            fatal_exception(x);
        } catch (...) {
            fatal_unknown_exception();
        }
        return v;
    }
};

void check_cuda() {
    if (0 < CudaProperties::getCudaDeviceCount())
        return;

    stringstream ss;
    ss   << "Sonic AWE requires you to have installed graphics drivers from NVIDIA." << endl
         << endl
         << "You need to have one of these graphics cards from NVIDIA;" << endl
         << "   http://www.nvidia.com/object/cuda_gpus.html" << endl
         << endl
         << "You also need to have NVIDIAs display drivers installed;" << endl
         <<"    http://www.nvidia.com" << endl
         << endl
         << endl
         << "Sonic AWE cannot start." << endl;
    cerr << ss.str();
    cerr.flush();

    QMessageBox::critical( 0,
                 "Cannot start Sonic AWE",
                 QString::fromStdString(ss.str()) );

    exit(-1);
}
int main(int argc, char *argv[])
{

    QDateTime now = QDateTime::currentDateTime();
    now.date().year();
    stringstream ss;
    ss << "Sonic Awe - ";
#ifdef SONICAWE_VERSION
    ss << TOSTR(SONICAWE_VERSION);
#else
    ss << __DATE__;// << " - " << __TIME__;
#endif

#ifdef SONICAWE_BRANCH
    if( 0 < strlen( TOSTR(SONICAWE_BRANCH) ))
        ss << " - branch: " << TOSTR(SONICAWE_BRANCH);
#endif

    _sawe_version_string = ss.str();

    SonicAWE_Application a(argc, argv);
    check_cuda();

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

    if (0 == _soundfile.length() || !QFile::exists(_soundfile.c_str())) {
    	QString fileName = QFileDialog::getOpenFileName(0, "Open sound file");
        if (0 == fileName.length())
            return 0;
        _soundfile = fileName.toStdString();
    }
    printf("Reading file: %s\n", _soundfile.c_str());

    switch ( _yscale )
    {
        case DisplayWidget::Yscale_Linear:
        case DisplayWidget::Yscale_ExpLinear:
        case DisplayWidget::Yscale_LogLinear:
        case DisplayWidget::Yscale_LogExpLinear:
            break;
        default:
            printf("Invalid yscale value, must be one of {1, 2, 3, 4}\n\n%s", _sawe_usage_string);
            exit(1);
    }

    try {
        boost::shared_ptr<Waveform> wf( new Waveform( _soundfile.c_str() ) );
        boost::shared_ptr<Transform> wt( new Transform(wf, _channel, _samples_per_chunk, _scales_per_octave, _wavelet_std_t ) );

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

        w.setCentralWidget( dw.get() );
        dw->show();
        w.show();

       return a.exec();
    } catch (const std::exception &x) {
        fatal_exception(x);
    } catch (...) {
        fatal_unknown_exception();
    }
}

