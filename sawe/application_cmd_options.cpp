#include "application.h"

// heightmap
#include "heightmap/renderer.h"

// tfr
#include "tfr/cwt.h"

// adapters
#include "adapters/csv.h"
#include "adapters/hdf5.h"
#include "adapters/playback.h"

// boost
#include <boost/foreach.hpp>

// std
#include <string>
#include <stdio.h>

using namespace std;

namespace Sawe {

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
    "    wavelet_time_support Transform CWT chunks with this many sigmas overlap in\n"
    "                       time domain.\n"
    "    wavelet_scale_support Transform CWT chunks with this many sigmas overlap in\n"
    "                       scale domain.\n"
    "    record             Starts Sonic AWE starts in record mode. [default]\n"
    "    record_device      Selects a specific device for recording. -1 specifices\n"
    "                       the default input device/microphone.\n"
    "    playback_device    Selects a specific device for playback. -1 specifices the\n"
    "                       default output device.\n"
    "    list_audio_devices Lists all available audio devices that Sonic AWE can use.\n"
/*    "    multithread        If set, starts a parallell worker thread. Good if heavy \n"
    "                       filters are being used as the GUI won't be locked during\n"
    "                       computation.\n"
*/    "\n"
    "Sonic AWE, 2010\n";

static unsigned _channel=0;
static unsigned _scales_per_octave = 20;
static float _wavelet_time_support = 5;
static float _wavelet_scale_support = 4;
static unsigned _samples_per_chunk = 1;
static unsigned _samples_per_block = 1<<7;
static unsigned _scales_per_block = 1<<8;
static unsigned _get_hdf = (unsigned)-1;
static unsigned _get_csv = (unsigned)-1;
static bool _get_chunk_count = false;
static std::string _selectionfile = "selection.wav";
static bool _record = false;
static bool _list_audio_devices = false;
static int _record_device = -1;
static int _playback_device = -1;
static std::string _soundfile = "";
#ifndef QT_NO_THREAD
static bool _multithread = false;
#endif
static bool _sawe_exit = false;

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
        else if (readarg(&cmd, wavelet_scale_support));
        else if (readarg(&cmd, samples_per_block));
        else if (readarg(&cmd, scales_per_block));
        else if (readarg(&cmd, get_chunk_count));
        else if (readarg(&cmd, record_device));
        else if (readarg(&cmd, record));
        else if (readarg(&cmd, list_audio_devices));
        else if (readarg(&cmd, playback_device));
        else if (readarg(&cmd, channel));
        else if (readarg(&cmd, get_hdf));
        else if (readarg(&cmd, get_csv));
#ifndef QT_NO_THREAD
        else if (readarg(&cmd, multithread));
#endif
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


void Application::
        parse_command_line_options(int& argc, char **argv)
{
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
                ::exit(1);
            }
            argv++;
            argc--;
        }
    }

    Sawe::pProject p; // p will be owned by Application and released before a.exec()

    if (!_soundfile.empty())
        p = Sawe::Project::open( _soundfile );

    if (!p)
        p = Sawe::Project::createRecording( _record_device );

    if (!p)
        ::exit(-1);

    this->openadd_project( p );

    p->mainWindow(); // Ensures that an OpenGL context is created

    BOOST_ASSERT( QGLContext::currentContext() );

    apply_command_line_options( p );

    Tfr::Cwt& cwt = Tfr::Cwt::Singleton();
    unsigned total_samples_per_chunk = cwt.prev_good_size( 1<<_samples_per_chunk, p->head_source()->sample_rate() );
    TaskTimer("Samples per chunk = %d", total_samples_per_chunk).suppressTiming();

    if (_get_csv != (unsigned)-1) {
        if (0==p->head_source()->number_of_samples()) {
            Sawe::Application::display_fatal_exception(std::invalid_argument("Can't extract CSV without input file."));
            ::exit(-1);
        }

        Adapters::Csv csv;
        csv.source( p->head_source() );
        csv.read( Signal::Interval( _get_csv*total_samples_per_chunk, (_get_csv+1)*total_samples_per_chunk ));
        _sawe_exit = true;
    }

    if (_get_hdf != (unsigned)-1) {
        if (0==p->head_source()->number_of_samples()) {
            Sawe::Application::display_fatal_exception(std::invalid_argument("Can't extract HDF without input file."));
            ::exit(-1);
        }

        Adapters::Hdf5Chunk hdf5;
        hdf5.source( p->head_source() );
        hdf5.read( Signal::Interval(_get_hdf*total_samples_per_chunk, (_get_hdf+1)*total_samples_per_chunk ));
        _sawe_exit = true;
    }

    if (_get_chunk_count != false) {
        cout << p->head_source()->number_of_samples() / total_samples_per_chunk << endl;
        _sawe_exit = true;
    }

    if (_list_audio_devices)
    {
        Adapters::Playback::list_devices();
        _sawe_exit = true;
    }

    if (_sawe_exit)
    {
        ::exit(0);
    }
}

void Application::
        apply_command_line_options( pProject p )
{

    Tfr::Cwt& cwt = Tfr::Cwt::Singleton();
    cwt.scales_per_octave( _scales_per_octave );
    cwt.wavelet_time_support( _wavelet_time_support );
    cwt.wavelet_scale_support( _wavelet_scale_support );

    //p->worker.samples_per_chunk_hint( _samples_per_chunk );
#ifndef QT_NO_THREAD
    if (_multithread)
        p->worker.start();
#endif

    Tools::ToolFactory &tools = p->tools();

    tools.playback_model.playback_device = _playback_device;
    tools.playback_model.selection_filename  = _selectionfile;

    BOOST_FOREACH( const boost::shared_ptr<Heightmap::Collection>& c, tools.render_model.collections )
    {
        c->samples_per_block( _samples_per_block );
        c->scales_per_block( _scales_per_block );
    }
}

} // namespace Sawe
