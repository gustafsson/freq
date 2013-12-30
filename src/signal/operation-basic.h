#ifndef SIGNALOPERATIONBASIC_H
#define SIGNALOPERATIONBASIC_H

#include "signal/operation.h"

namespace Signal {

/**
  Example 1:
  start:  1234567
  OperationSetSilent( start, 1, 2 );
  result: 1004567
*/
class OperationSetSilent: public Signal::OperationDesc {
public:
    class Operation: public Signal::Operation {
    public:
        Operation( const Signal::Interval& section );

        Signal::pBuffer process(Signal::pBuffer b);

        Signal::Interval section() { return section_; }

    private:
        Signal::Interval section_;
    };

    OperationSetSilent( const Signal::Interval& section );

    // OperationDesc
    Interval requiredInterval( const Interval& I, Interval* expectedOutput ) const;
    Interval affectedInterval( const Interval& I ) const;
    OperationDesc::Ptr copy() const;
    Operation::Ptr createOperation(ComputingEngine* engine=0) const;
    QString toString() const;

    Signal::Interval section() { return section_; }
private:
    Signal::Interval section_;


    friend class boost::serialization::access;
    OperationSetSilent():section_(0,0) {} // only used by deserialization

    template<class archive> void serialize(archive& ar, const unsigned int /*version*/)
    {
        using boost::serialization::make_nvp;

        ar & BOOST_SERIALIZATION_NVP(section_.first)
           & BOOST_SERIALIZATION_NVP(section_.last);
    }
};

} // namespace Signal

#endif // SIGNALOPERATIONBASIC_H
