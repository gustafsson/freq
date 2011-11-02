#ifndef CSVTIMESERIES_H
#define CSVTIMESERIES_H

#include "signal/buffersource.h"
#include "sawe/reader.h"

#include <boost/serialization/string.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/identity.hpp>

#ifdef _MSC_VER
typedef unsigned __int64 uint64_t;
typedef unsigned __int32 uint32_t;
#else
#include <stdint.h>
#endif

#include "cpumemorystorage.h"

namespace Adapters {

class CsvTimeseries : public Signal::BufferSource
{
private:
    template<class Archive>
    struct save_binary_type {
        static void invoke(
            Archive & ar,
            const std::vector<char> & data
        ){
            unsigned N = data.size();
            ar & boost::serialization::make_nvp( "N", N );
            ar.save_binary( &data[0], N );
        }
    };

    template<class Archive>
    struct load_binary_type {
        static void invoke(
            Archive & ar,
            std::vector<char> & data
        ){
            unsigned N = 0;
            ar & boost::serialization::make_nvp( "N", N );
            data.resize(N);
            ar.load_binary( &data[0], N );
        }
    };

public:
    static std::string getFileFormatsQtFilter( bool split );

    CsvTimeseries(std::string filename);

    virtual std::string name();
    std::string filename() const { return _original_relative_filename; }

private:
    CsvTimeseries() {} // for deserialization
    void load(std::string filename );

    std::string _original_relative_filename;

    std::vector<char> rawdata;
    static std::vector<char> getRawFileData(std::string filename);
    void load(std::vector<char> rawFileData);

    friend class boost::serialization::access;
    template<class archive> void serialize(archive& ar, const unsigned int /*version*/) {
        using boost::serialization::make_nvp;

        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Operation);

        ar & make_nvp("Original_filename", _original_relative_filename);

        typedef BOOST_DEDUCED_TYPENAME boost::mpl::eval_if<
            BOOST_DEDUCED_TYPENAME archive::is_saving,
            boost::mpl::identity<save_binary_type<archive> >,
            boost::mpl::identity<load_binary_type<archive> >
        >::type typex;
        typex::invoke(ar, rawdata);
        //ar & make_nvp("Rawdata", rawdata);

        if (typename archive::is_loading())
            load( rawdata );

        uint64_t X = 0;
        for (unsigned c=0; c<_waveforms.size(); ++c)
        {
            unsigned char* p = (unsigned char*)CpuMemoryStorage::ReadOnly( _waveforms[c]->waveform_data() ).ptr();
            unsigned N = _waveforms[c]->waveform_data()->numberOfBytes();

            for (unsigned i=0; i<N; ++i)
            {
                X = *p*X + *p;
                ++p;
            }
        }

        const uint32_t checksum = (uint32_t)X;
        uint32_t V = checksum;
        ar & make_nvp("V", V);

#if defined(TARGET_reader)
        if (checksum != V)
        {
            throw std::ios_base::failure("Sonic AWE Reader can only open original files");
        }
#endif
    }
};


} // namespace Adapters

#endif // ADAPTERS_CSVTIMESERIES_H
