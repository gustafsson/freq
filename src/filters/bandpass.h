#ifndef BANDPASS_H
#define BANDPASS_H

#include "tfr/stftfilter.h"

namespace Filters {

class Bandpass: public Tfr::StftFilter
{
public:
    Bandpass(float f1, float f2, bool save_inside=false);

    virtual std::string name();
    virtual bool operator()( Tfr::Chunk& );

    float _f1, _f2;
    bool _save_inside;

private:
    Bandpass() {} // for deserialization

    friend class boost::serialization::access;
    template<class archive> void serialize(archive& ar, const unsigned int /*version*/) {
        using boost::serialization::make_nvp;

        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Operation)
           & make_nvp("f1", _f1) & make_nvp("f2", _f1)
           & make_nvp("save_inside", _save_inside);
    }
};

} // namespace Filters

#endif // BANDPASS_H
