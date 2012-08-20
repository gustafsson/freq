#include "sawe/configuration.h"

#include "sawe/reader.h"

#if !defined(_DEBUG) || defined(DEBUG_WITH_OMP)
#include <omp.h>
#endif

#include <sstream>
#include <iostream>

#include <QSysInfo>
#include <QString>
#include <QSettings>

#ifdef Q_OS_LINUX
#include <QProcess>
#endif

#ifdef USE_OPENCL
#include "openclcontext.h"
#elif defined(USE_CUDA)
#include "CudaProperties.h"
#endif


#define STRINGIFY(x) #x
#define TOSTR(x) STRINGIFY(x)

using namespace std;

namespace Sawe
{

Configuration::
        Configuration()
            :
            skip_update_check_( false ),
            use_saved_state_( true ),
            channel_( 0 ),
            scales_per_octave_( 27.f+2.f/3 ), // this gives an stft window size of 4096 samples (with default cwt settings) due to the combined slider in RenderController::receiveSetTimeFrequencyResolution
            wavelet_time_support_( 3 ),
            wavelet_scale_support_( 3 ),
            min_hz_( 60 ),
            samples_per_chunk_hint_( 1 ),
            samples_per_block_( 1<<8 ),
            scales_per_block_( 1<<8 ),
            get_hdf_( (unsigned)-1 ),
            get_csv_( (unsigned)-1 ),
            get_chunk_count_( false ),
            selectionfile_( "selection.wav" ),
            soundfile_( "" ),
            multithread_( false )
{
#ifdef SONICAWE_VERSION
    version_ = TOSTR(SONICAWE_VERSION);
#endif


#ifdef SONICAWE_BRANCH
    branch_ = TOSTR(SONICAWE_BRANCH);
#endif


#ifdef SONICAWE_REVISION
    revision_ = TOSTR(SONICAWE_REVISION);
#endif


#ifdef SONICAWE_UNAME
    uname_ = TOSTR(SONICAWE_UNAME);
    #ifdef SONICAWE_UNAMEm
        uname_ += " " TOSTR(SONICAWE_UNAMEm);
    #endif
    #ifdef SONICAWE_DISTCODENAME
        uname_ += " " TOSTR(SONICAWE_DISTCODENAME);
    #endif
#else
    #ifdef _DEBUG
        uname_ = "debug build, undefined platform";
    #else
        uname_ = "undefined platform";
    #endif
#endif


#ifdef USE_CUDA
    computing_platform_ = "CUDA";
#elif defined(USE_OPENCL)
    computing_platform_ = "OPENCL";
#else
    computing_platform_ = "CPU";
#endif


    uname_ += " ";
    uname_ += computing_platform_;


#ifdef SAWE_MONO
    mono_ = true;
#else
    mono_ = false;
#endif
}


string Configuration::
        version()
{
    return Singleton().version_;
}


string Configuration::
        branch()
{
    return Singleton().branch_;
}


string Configuration::
        revision()
{
    return Singleton().revision_;
}


string Configuration::
        uname()
{
    return Singleton().uname_;
}


string Configuration::
        distcodename()
{
    return Singleton().distcodename_;
}


std::string Configuration::
        build_date()
{
    return __DATE__;
}


std::string Configuration::
        build_time()
{
    return __TIME__;
}


bool Configuration::
        mono()
{
    return Singleton().mono_;
}


string Configuration::
        version_string()
{
    if (Singleton().version_string_.empty())
        rebuild_version_string();
    return Singleton().version_string_;
}


string Configuration::
        title_string()
{
    if (Singleton().title_string_.empty())
        rebuild_version_string();
    return Singleton().title_string_;
}


string Configuration::
        operatingSystemName()
{
    stringstream name;
    name << operatingSystemNameWithoutBits() << " " << (sizeof(void*)<<3) << "-bit";
    return name.str();
}


string Configuration::
        operatingSystemNameWithoutBits()
{
#ifdef Q_WS_WIN
    switch(QSysInfo::WindowsVersion)
    {
    case 0: return "unknown Windows version";
    case QSysInfo::WV_32s: return "Windows 3.1";
    case QSysInfo::WV_95: return "Windows 95";
    case QSysInfo::WV_98: return "Windows 98";
    case QSysInfo::WV_Me: return "Windows Me";
    case QSysInfo::WV_NT: return "Windows NT";
    case QSysInfo::WV_2000: return "Windows 2000";
    case QSysInfo::WV_XP: return "Windows XP";
    case QSysInfo::WV_2003: return "Windows Server 2003 (or in the same family)";
    case QSysInfo::WV_VISTA: return "Windows Vista";
    case QSysInfo::WV_WINDOWS7: return "Windows 7";
    case QSysInfo::WV_NT_based: return "Windows 8 (or otherwise NT-based)";
    default: return QString("unrecognized Windows version (%1)").arg(QSysInfo::WindowsVersion).toStdString();
    }
#endif
#ifdef Q_OS_MAC
#define xMV_10_8 (QSysInfo::MV_10_7+1)
    switch(QSysInfo::MacintoshVersion)
    {
    case QSysInfo::MV_Unknown: return "unknown Mac version";
    case QSysInfo::MV_9: return "Mac OS X 9";
    case QSysInfo::MV_10_0: return "Mac OS X 10.0 (Cheetah)";
    case QSysInfo::MV_10_1: return "Mac OS X 10.1 (Puma)";
    case QSysInfo::MV_10_2: return "Mac OS X 10.2 (Jaguar)";
    case QSysInfo::MV_10_3: return "Mac OS X 10.3 (Panther)";
    case QSysInfo::MV_10_4: return "Mac OS X 10.4 (Tiger)";
    case QSysInfo::MV_10_5: return "Mac OS X 10.5 (Leopard)";
    case QSysInfo::MV_10_6: return "Mac OS X 10.6 (Snow Leopard)";
    case QSysInfo::MV_10_7: return "OS X 10.7 (Lion)";
    case xMV_10_8: return "OS X 10.8 (Mountain Lion)";
    default: return QString("unrecognized Mac OS X version (%1)").arg(QSysInfo::MacintoshVersion).toStdString();
    }
#endif
#ifdef Q_OS_LINUX
    QString codenamecmd = "lsb_release -c";
    QString descriptioncmd = "lsb_release -d";

    QString description = "unknown Linux distribution";
    QString codename = "";

    QProcess process;

    process.start(descriptioncmd);
    process.waitForFinished();
    if (QProcess::NormalExit == process.exitStatus() && 0 == process.exitCode())
        description = process.readAllStandardOutput();

    process.start(codenamecmd);
    process.waitForFinished();
    if (QProcess::NormalExit == process.exitStatus() && 0 == process.exitCode())
        codename = process.readAllStandardOutput();

    QRegExp discardlabel(".*\t(.*)\n");
    description = description.replace(discardlabel,"\\1");
    codename = codename.replace(discardlabel,"\\1");

    if (!codename.isEmpty())
        description += " (" + codename + ")";

    return description.toStdString();
#endif
}


Configuration::OperatingSystemFamily Configuration::
        operatingSystemFamily()
{
#ifdef Q_WS_WIN
    return OperatingSystemFamily_Windows;
#endif
#ifdef Q_OS_MAC
    return OperatingSystemFamily_Mac;
#endif
#ifdef Q_OS_LINUX
    return OperatingSystemFamily_Ubuntu;
#endif
}


std::string Configuration::
        operatingSystemFamilyName()
{
    switch(operatingSystemFamily())
    {
    case OperatingSystemFamily_Windows: return "win";
    case OperatingSystemFamily_Mac: return "mac";
    case OperatingSystemFamily_Ubuntu: return "ubuntu";
    default: return "unknown";
    }
}

int Configuration::
        cpuCores()
{
#if !defined(_DEBUG) || defined(DEBUG_WITH_OMP)
    return omp_get_max_threads();
#else
    return 1;
#endif
}


string Configuration::
        computationDeviceName()
{
#ifdef USE_OPENCL
    return "OpenCL on " + OpenCLContext::Singleton().deviceName();
#elif defined(USE_CUDA)
    stringstream cuda;
    unsigned gigaflops = CudaProperties::flops(CudaProperties::getCudaDeviceProp()) / 1000000000;
    cuda << "Cuda on " << CudaProperties::getCudaDeviceProp().name << " with " << gigaflops << " gigaflops";
    return cuda.str();
#else
    stringstream cpu;
    int cores = cpuCores();
    cpu << "CPU with " << cores << " core" << (cores==1?"":"s");
    return cpu.str();
#endif
}


Configuration::DeviceType Configuration::
        computationDeviceType()
{
#ifdef USE_OPENCL
    return DeviceType_OpenCL;
#elif defined(USE_CUDA)
    return DeviceType_Cuda;
#else
    return DeviceType_CPU;
#endif
}


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
    "Generic settings\n"
    "    --mono=1            Makes Sonic AWE only process the first channel\n"
    "    --use_saved_state=0 Disables restoring old user interface states\n"
    "    --skip_update_check=1 Disables checking for new versions\n"
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

//    "    multithread        If set, starts a parallell worker thread. Good if heavy \n"
//    "                       filters are being used as the GUI won't be locked during\n"
//    "                       computation.\n"

    "\n"
    "Sonic AWE is a product developed by MuchDifferent\n";

stringstream commandline_message_;


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
void atoval(const char *cmd, string& val) {
    val = cmd;
}

#define readarg(cmd, name) tryreadarg(cmd, "--"#name, #name, name##_)

template<typename Type>
bool tryreadarg(const char **cmd, const char* prefix, const char* name, Type &val) {
    if (0 != prefixcmp(*cmd, prefix))
        return 0;
    *cmd += strlen(prefix);
    if (**cmd == '=')
        atoval(*cmd+1, val);
    else if (**cmd != 0)
        return 0;
    else
        commandline_message_ << "default " << name << "=" << val << endl;

    return 1;
}

template<>
bool tryreadarg(const char **cmd, const char* prefix, const char*, bool &val) {
    if (prefixcmp(*cmd, prefix))
        return 0;
    *cmd += strlen(prefix);
    if (**cmd == '=')
        atoval(*cmd+1, val);
    else if (**cmd != 0)
        return 0;
    else
        val = true;

    return 1;
}


int Configuration::
    handle_options(char ***argv, int *argc)
{
    int handled = 0;

    while (*argc > 0) {
        const char *cmd = (*argv)[0];
        if (cmd[0] != '-')
            break;

        if (!strcmp(cmd, "--help")) {
            commandline_message_  << "See the logfile sonicawe.log for a list of valid command line options.";
        } else if (!strcmp(cmd, "--version")) {
            commandline_message_ << version_string().c_str();
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
        else if (readarg(&cmd, version));
        else if (readarg(&cmd, use_saved_state));
        else if (readarg(&cmd, skip_update_check));
#ifndef QT_NO_THREAD
        else if (readarg(&cmd, multithread));
#endif
        // TODO use _selectionfile
        else {
            commandline_message_ << "Unknown option: " << cmd << endl
                    << "See the logfile sonicawe.log for a list of valid command line options.";
            break;
        }

        (*argv)++;
        (*argc)--;
        handled++;
    }

    return handled;
}


string Configuration::
        parseCommandLineOptions( int& argc, char* argv[] )
{
    // skip application filename
    argv++;
    argc--;

    while (argc) {
        Singleton().handle_options(&argv, &argc);

        if (!commandline_message_.str().empty())
            break;

        if (argc) {
            if (Singleton().soundfile_.empty()) {
                Singleton().soundfile_ = argv[0];
            } else {
                commandline_message_
                        << "Unknown command line option: " << argv[0] << endl
                        << "Sonic AWE takes only one file as input argument. Will try to open \""
                        << Singleton().soundfile_ << "\"" << endl
                        << endl
                        << "See the logfile sonicawe.log for a list of valid command line options.";
                cerr << commandline_message_.str() << endl;
                cerr << _sawe_usage_string << endl;
                break;
            }
            argv++;
            argc--;
        }
    }

    return parseCommandLineMessage();
}


std::string Configuration::
        parseCommandLineMessage()
{
    return commandline_message_.str();
}


string Configuration::
        commandLineUsageString()
{
    return _sawe_usage_string;
}


string Configuration::
        input_file()
{
    return Singleton().soundfile_;
}


string Configuration::
        selection_output_file()
{
    return Singleton().selectionfile_;
}


void Configuration::
        rebuild_version_string()
{
    stringstream ss;
    if (!Sawe::Configuration::version().empty())
        ss << "v" << Sawe::Configuration::version();
    else
    {
        ss << "dev " << build_date();
        #ifdef _DEBUG
            ss << ", " << build_time();
        #endif

        if (!Sawe::Configuration::branch().empty())
            ss << " - branch: " << Sawe::Configuration::branch();
    }

    Singleton().version_string_ = ss.str();
    Singleton().title_string_ = Reader::reader_title() + " - " + Singleton().version_string_;
}


unsigned Configuration::
        samples_per_chunk_hint()
{
    return Singleton().samples_per_chunk_hint_;
}


unsigned Configuration::
        get_hdf()
{
    return Singleton().get_hdf_;
}


unsigned Configuration::
        get_csv()
{
    return Singleton().get_hdf_;
}


bool Configuration::
        get_chunk_count()
{
    return Singleton().get_chunk_count_;
}


float Configuration::
        scales_per_octave()
{
    return Singleton().scales_per_octave_;
}


float Configuration::
        wavelet_time_support()
{
    return Singleton().wavelet_time_support_;
}


float Configuration::
        wavelet_scale_support()
{
    return Singleton().wavelet_scale_support_;
}


float Configuration::
        min_hz()
{
    return Singleton().min_hz_;
}


unsigned Configuration::
        samples_per_block()
{
    return Singleton().samples_per_block_;
}


unsigned Configuration::
        scales_per_block()
{
    return Singleton().scales_per_block_;
}


bool Configuration::
        skip_update_check()
{
    return Singleton().skip_update_check_;
}


bool Configuration::
        use_saved_state()
{
    return Singleton().use_saved_state_;
}


void Configuration::
        resetDefaultSettings()
{
#define DO_EXPAND(VAL)  VAL ## 1
#define EXPAND(VAL)     DO_EXPAND(VAL)

#if !defined(TARGETNAME) || (EXPAND(TARGETNAME) == 1)
    QSettings().setValue("target","");
#else
    QSettings().setValue("target",TOSTR(TARGETNAME));
#endif
}


} // namespace Sawe
