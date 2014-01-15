#ifndef NORMALIZE_H
#define NORMALIZE_H

#include "signal/operation.h"

#include <boost/serialization/nvp.hpp>

namespace Filters {

/**
 * @brief The Normalize class should normalize the signal strength.
 */
class Normalize: public Signal::OperationDesc
{
public:
    enum Method
    {
        Method_None,
        Method_InfNorm,
        Method_2Norm,
        Method_TruncatedMean
    };

    Normalize( unsigned normalizationRadius, Method method = Method_InfNorm );

    Signal::Interval requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const;
    Signal::Interval affectedInterval( const Signal::Interval& I ) const;
    Signal::OperationDesc::Ptr copy() const;
    Signal::Operation::Ptr createOperation(Signal::ComputingEngine* engine) const;
    QString toString() const;

    unsigned radius();

private:
    Normalize(); // used by deserialization
    unsigned normalizationRadius;

    friend class boost::serialization::access;
    template<class archive> void serialize(archive& ar, const unsigned int) {
        using boost::serialization::make_nvp;

        ar & BOOST_SERIALIZATION_NVP(normalizationRadius);
    }

public:
    static void test();
};

} // namespace Filters
#endif // NORMALIZE_H
