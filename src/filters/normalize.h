#ifndef NORMALIZE_H
#define NORMALIZE_H

#include "signal/operation.h"
namespace Filters {

class Normalize: public Signal::DeprecatedOperation
{
public:
    enum Method
    {
        Method_None,
        Method_Standard,
        Method_TruncatedMean
    };

    Normalize( unsigned normalizationRadius, Signal::pOperation source = Signal::pOperation() );

    virtual std::string name();
    virtual void invalidate_samples(const Signal::Intervals& I);
    virtual Signal::pBuffer read( const Signal::Interval& I );

private:
    Normalize(); // used by deserialization
    unsigned normalizationRadius;

    friend class boost::serialization::access;
    template<class archive> void serialize(archive& ar, const unsigned int) {
        using boost::serialization::make_nvp;

        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(DeprecatedOperation);

        ar & BOOST_SERIALIZATION_NVP(normalizationRadius);
    }
};

} // namespace Filters
#endif // NORMALIZE_H
