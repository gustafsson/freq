#include "application.h"

// heightmap
#include "heightmap/renderer.h"

// tfr
#include "tfr/cwt.h"

// adapters
#include "adapters/csv.h"
#include "adapters/hdf5.h"
#include "adapters/playback.h"

// gpumisc
#include <redirectstdout.h>

// boost
#include <boost/foreach.hpp>

// Qt
#include <QGLContext>
#include <QMessageBox>

// std
#include <string>
#include <stdio.h>

using namespace std;

namespace Sawe {

static const char _sawe_usage_string[] =
    "\n"
    "sonicawe [--parameter=value]* [FILENAME]\n"
    "sonicawe [--parameter] \n"
    "sonicawe [--help] \n"
    "sonicawe [--version] \n"
    "\n"
    "    Sonic AWE is a signal analysis tool based on visualization.\n"
    "\n"
    "    By using command line parameters it's possible to let Sonic AWE compute\n"
    "    a Continious Gabor Wavelet Transform (CWT below). Each parameter takes a\n"
    "    number as value, if no value is given the default value is written to \n"
    "    standard output and the program exits immediately after. Valid parameters \n"
    "    are:\n"
    "\n"
    "Ways of extracting data from a Continious Gabor Wavelet Transform (CWT)\n"
    "    --get_csv=number    Saves the given chunk number into sawe.csv which \n"
    "                        then can be read by matlab or octave.\n"
    "    --get_hdf=number    Saves the given chunk number into sawe.h5 which \n"
    "                        then can be read by matlab or octave.\n"
    "    --get_chunk_count=1 outpus the number of chunks that can be fetched by \n"
    "                        the --get_* options\n"
    "\n"
    "Settings for computing CWT\n"
    "    --samples_per_chunk_hint\n"
    "                        The transform is computed in chunks from the input\n"
    "                        This value determines the number of input samples that\n"
    "                        should correspond to one chunk of the transform by \n"
    "                        2^samples_per_chunk_hint. The actual number of samples\n"
    "                        computed and written to file per chunk might be \n"
    "                        different.\n"
    "    --scales_per_octave Frequency accuracy of CWT, higher accuracy takes more\n"
    "                        time to compute.\n"
    "    --wavelet_time_support\n"
    "                        Transform CWT chunks with this many sigmas overlap in time\n"
    "                        domain.\n"
    "    --wavelet_scale_support\n"
    "                        Transform CWT chunks with this many sigmas overlap in scale\n"
    "                        domain.\n"
    "    --min_hz            Transform CWT with scales logarithmically distributed\n"
    "                        on (min_hz, fs/2]\n"
    "\n"
    "Rendering settings\n"
    "    --samples_per_block The transform chunks are downsampled to blocks for\n"
    "                        rendering, this gives the number of samples per block.\n"
    "    --scales_per_block  Number of scales per block, se samples_per_block.\n"

/*    "    multithread        If set, starts a parallell worker thread. Good if heavy \n"
    "                       filters are being used as the GUI won't be locked during\n"
    "                       computation.\n"
*/    "\n"
    "Sonic AWE, 2011\n";

static unsigned _channel=0;
static unsigned _scales_per_octave = 20;
static float _wavelet_time_support = 5;
static float _wavelet_scale_support = 4;
static float _min_hz = 20;
static unsigned _samples_per_chunk_hint = 1;
static unsigned _samples_per_block = 1<<8;
static unsigned _scales_per_block = 1<<8;
static unsigned _get_hdf = (unsigned)-1;
static unsigned _get_csv = (unsigned)-1;
static bool _get_chunk_count = false;
static std::string _selectionfile = "selection.wav";
static std::string _soundfile = "";
#ifndef QT_NO_THREAD
static bool _multithread = false;
#endif


std::stringstream message;


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
        message << "default " << name << "=" << val << endl;
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
            message  << "See the logfile sonicawe.log for a list of valid command line options.";
        } else if (!strcmp(cmd, "--version")) {
            message << Sawe::Application::version_string().c_str();
        }
        else if (readarg(&cmd, samples_per_chunk_hint));
        else if (readarg(&cmd, scales_per_octave));
        else if (readarg(&cmd, wavelet_time_support));
        else if (readarg(&cmd, wavelet_scale_support));
        else if (readarg(&cmd, samples_per_block));
        else if (readarg(&cmd, scales_per_block));
        else if (readarg(&cmd, get_chunk_count));
        else if (readarg(&cmd, channel));
        else if (readarg(&cmd, get_hdf));
        else if (readarg(&cmd, get_csv));
        else if (readarg(&cmd, min_hz));
#ifndef QT_NO_THREAD
        else if (readarg(&cmd, multithread));
#endif
        // TODO use _selectionfile
        else {
            message << "Unknown option: " << cmd << endl
                    << "See the logfile sonicawe.log for a list of valid command line options.";
            break;
        }

        (*argv)++;
        (*argc)--;
        handled++;
    }

    return handled;
}


void Application::
        parse_command_line_options(int& argc, char **argv)
{
    // skip application filename
    argv++;
    argc--;

    while (argc) {
        handle_options(&argv, &argc);

        if (!message.str().empty())
            break;

        if (argc) {
            if (_soundfile.empty()) {
                _soundfile = argv[0];
            } else {
                std::stringstream ss;
                ss      << "Unknown command line option: " << argv[0] << endl
                        << "Sonic AWE takes only one file as input argument. Will try to open \""
                        << _soundfile << "\"" << endl
                        << endl
                        << "See the logfile sonicawe.log for a list of valid command line options.";
                cerr << ss.str() << endl;
                cerr << _sawe_usage_string << endl;
                QMessageBox::warning(0, "Sonic AWE", QString::fromStdString( ss.str()) );
                break;
            }
            argv++;
            argc--;
        }
    }

    if (!message.str().empty())
    {
        cerr    << message.str() << endl    // Want output in logfile
                << _sawe_usage_string;
        this->rs.reset();
        cerr    << message.str() << endl    // Want output in console window, if any
                << _sawe_usage_string;
        QMessageBox::critical(0, "Sonic AWE", QString::fromStdString( message.str()) );
        ::exit(-1);
        //mb.setWindowModality( Qt::ApplicationModal );
        //mb.show();

        return;
    }


    Sawe::pProject p; // p will be owned by Application and released before a.exec()

    if (!_soundfile.empty())
        p = Sawe::Application::slotOpen_file( _soundfile );

    if (!p)
        p = Sawe::Application::slotNew_recording( -1 );

    if (!p)
        ::exit(-1);

    Tfr::Cwt& cwt = Tfr::Cwt::Singleton();
    Signal::pOperation source = p->tools().render_model.renderSignalTarget->post_sink()->source();
    unsigned total_samples_per_chunk = cwt.prev_good_size( 1<<_samples_per_chunk_hint, source->sample_rate() );

    bool sawe_exit = false;

    if (_get_csv != (unsigned)-1) {
        if (0==source->number_of_samples()) {
            Sawe::Application::display_fatal_exception(std::invalid_argument("Can't extract CSV without input file."));
            ::exit(-1);
        }

        Adapters::Csv csv(QString("sonicawe-%1.csv").arg(_get_csv).toStdString());
        csv.source( source );
        csv.read( Signal::Interval( _get_csv*total_samples_per_chunk, (_get_csv+1)*total_samples_per_chunk ));
        TaskInfo("Samples per chunk = %u", total_samples_per_chunk);
        sawe_exit = true;
    }

    if (_get_hdf != (unsigned)-1) {
        if (0==source->number_of_samples()) {
            Sawe::Application::display_fatal_exception(std::invalid_argument("Can't extract HDF without input file."));
            ::exit(-1);
        }

        Adapters::Hdf5Chunk hdf5(QString("sonicawe-%1.h5").arg(_get_hdf).toStdString());
        hdf5.source( source );
        hdf5.read( Signal::Interval(_get_hdf*total_samples_per_chunk, (_get_hdf+1)*total_samples_per_chunk ));
        TaskInfo("Samples per chunk = %u", total_samples_per_chunk);
        sawe_exit = true;
    }

    if (_get_chunk_count != false) {
        TaskInfo("number of samples = %u", source->number_of_samples());
        TaskInfo("samples per chunk = %u", total_samples_per_chunk);
        TaskInfo("chunk count = %u", (source->number_of_samples() + total_samples_per_chunk-1) / total_samples_per_chunk);
        this->rs.reset();
        cout    << "number_of_samples = " << source->number_of_samples() << endl
                << "samples_per_chunk = " << total_samples_per_chunk << endl
                << "chunk_count = " << (source->number_of_samples() + total_samples_per_chunk-1) / total_samples_per_chunk << endl;
        sawe_exit = true;
    }

    if (sawe_exit)
    {
        ::exit(0);
    }
    else
    {
        // Ensures that an OpenGL context is created
        if( !QGLContext::currentContext() )
            QMessageBox::information(0,"Sonic AWE", "Sonic AWE couldn't start");
    }
}


void Application::
        apply_command_line_options( pProject p )
{
    Tfr::Cwt& cwt = Tfr::Cwt::Singleton();
    cwt.scales_per_octave( _scales_per_octave );
    cwt.wavelet_time_support( _wavelet_time_support );
    cwt.wavelet_scale_support( _wavelet_scale_support );
    //cwt.set_wanted_min_hz( _min_hz );

#ifndef SAWE_NO_MUTEX
    if (_multithread)
        p->worker.start();
#endif

    Tools::ToolFactory &tools = p->tools();

    tools.playback_model.selection_filename  = _selectionfile;

    BOOST_FOREACH( const boost::shared_ptr<Heightmap::Collection>& c, tools.render_model.collections )
    {
        c->samples_per_block( _samples_per_block );
        c->scales_per_block( _scales_per_block );
    }

    tools.render_view()->emitTransformChanged();
}

} // namespace Sawe
