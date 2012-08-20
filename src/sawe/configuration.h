#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include "sawedll.h"

#include <HasSingleton.h>
#include <sstream>
#include <vector>

namespace Sawe
{

class SaweDll Configuration: private HasSingleton<Configuration>
{
public:
    static std::string version();
    static std::string branch();
    static std::string revision();
    static std::string uname();
    static std::string distcodename();

    static std::string build_date();
    static std::string build_time();

    static bool mono();

    static std::string version_string();
    static std::string title_string();
    static void rebuild_version_string();

    /**
      ex:
      Windows 7 32-bit
      Ubuntu 10.04.3 LTS (lucid) 64-bit
      Mac OS X 10.6 (Snow Leopard) 32-bit
      */
    static std::string operatingSystemName();

    /**
      ex:
      Windows 7
      Ubuntu 10.04.3 LTS (lucid)
      Mac OS X 10.6 (Snow Leopard)
      */
    static std::string operatingSystemNameWithoutBits();

    enum OperatingSystemFamily
    {
        OperatingSystemFamily_Windows = 0x1,
        OperatingSystemFamily_Ubuntu = 0x2,
        OperatingSystemFamily_Mac = 0x3
    };
    static OperatingSystemFamily operatingSystemFamily();

    /**
      ex:
      win
      ubuntu
      mac
      */
    static std::string operatingSystemFamilyName();

    enum DeviceType
    {
        DeviceType_Cuda = 0x1,
        DeviceType_OpenCL = 0x2,
        DeviceType_GPUmask = 0x3,
        DeviceType_CPU = 0x4
    };

    static int cpuCores();
    static std::string computationDeviceName();
    static DeviceType computationDeviceType();


    static std::string parseCommandLineOptions( int& argc, char* argv[] );
    static std::string parseCommandLineMessage();
    static std::string commandLineUsageString();
    static std::string input_file();
    static std::string selection_output_file();

    static unsigned samples_per_chunk_hint();
    static unsigned get_hdf();
    static unsigned get_csv();
    static bool get_chunk_count();

    static float scales_per_octave();
    static float wavelet_time_support();
    static float wavelet_scale_support();
    static float min_hz();

    static unsigned samples_per_block();
    static unsigned scales_per_block();

    static bool skip_update_check();
    static bool use_saved_state();

    static bool feature(const std::string&);

    static void resetDefaultSettings();

private:
    friend class HasSingleton<Configuration>;
    Configuration();

    std::string version_;
    std::string branch_;
    std::string revision_;
    bool mono_;
    bool skip_update_check_;
    bool use_saved_state_;

    std::string uname_;
    std::string computing_platform_;
    std::string distcodename_;

    unsigned channel_;
    float scales_per_octave_;
    float wavelet_time_support_;
    float wavelet_scale_support_;
    float min_hz_;
    unsigned samples_per_chunk_hint_;
    unsigned samples_per_block_;
    unsigned scales_per_block_;
    unsigned get_hdf_;
    unsigned get_csv_;
    bool get_chunk_count_;
    std::string selectionfile_;
    std::string soundfile_;
    bool multithread_;

    std::string title_string_;
    std::string version_string_;
    std::vector<std::string> features_;

    int handle_options(char ***argv, int *argc);

    void set_basic_features();
    void set_default_features(const std::string&style="");
};

} // namespace Sawe

#endif // CONFIGURATION_H
