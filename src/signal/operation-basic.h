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
class OperationSetSilent: public Operation {
public:
    OperationSetSilent( Signal::pOperation source, const Signal::Interval& section );

    virtual pBuffer read( const Interval& I );
    virtual std::string name();

    virtual Signal::Intervals zeroed_samples() { return affected_samples(); }
    virtual Signal::Intervals affected_samples() { return section(); }

    virtual Signal::Interval section() { return section_; }
private:
    Signal::Interval section_;


    friend class boost::serialization::access;
    OperationSetSilent():Operation(pOperation()),section_(0,0) {} // only used by deserialization

    template<class archive> void serialize(archive& ar, const unsigned int /*version*/)
    {
        using boost::serialization::make_nvp;

        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Operation)
           & BOOST_SERIALIZATION_NVP(section_.first)
           & BOOST_SERIALIZATION_NVP(section_.last);
    }
};


class OperationRemoveSection: public Operation
{
public:
    OperationRemoveSection( pOperation source, Interval section );

    virtual pBuffer read( const Interval& I );
    virtual IntervalType number_of_samples();

    virtual Intervals affected_samples();
    virtual Intervals translate_interval(Intervals I);
    Intervals translate_interval_inverse(Intervals I);

private:

    Interval section_;

    friend class boost::serialization::access;
    OperationRemoveSection():Operation(pOperation()),section_(0,0) {} // only used by deserialization

    template<class archive> void serialize(archive& ar, const unsigned int /*version*/)
    {
        using boost::serialization::make_nvp;

        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Operation)
           & BOOST_SERIALIZATION_NVP(section_.first)
           & BOOST_SERIALIZATION_NVP(section_.last);
    }
};


/**
  Has no effect as long as source()->number_of_samples <= section.first.
  */
class OperationInsertSilence: public Operation
{
public:
    OperationInsertSilence( pOperation source, Interval section );

    virtual pBuffer read( const Interval& I );
    virtual IntervalType number_of_samples();

    virtual Intervals affected_samples();
    virtual Intervals translate_interval(Intervals I);
    Intervals translate_interval_inverse(Intervals I);
private:
    Interval section_;
};

class OperationSuperposition: public Operation
{
public:
    OperationSuperposition( pOperation source, pOperation source2 );

    virtual std::string name();
    void name(std::string);

    virtual pBuffer read( const Interval& I );

    virtual IntervalType number_of_samples();

    virtual unsigned num_channels();
    virtual void set_channel(unsigned c);
    virtual unsigned get_channel();

    virtual Intervals zeroed_samples();
    virtual Intervals affected_samples();
    virtual Intervals zeroed_samples_recursive();

    virtual pOperation source2() const { return _source2; }

    static pBuffer superPosition( pBuffer a, pBuffer b );
private:
    pOperation _source2;
    std::string _name;

    friend class boost::serialization::access;
    OperationSuperposition():Operation(pOperation()) {} // only used by deserialization

    template<class archive>
    void serialize(archive& ar, const unsigned int /*version*/)
    {
        using boost::serialization::make_nvp;

        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Operation)
           & BOOST_SERIALIZATION_NVP(_source2);
    }
};


class OperationAddChannels: public Operation, public boost::noncopyable
{
public:
    OperationAddChannels( pOperation source, pOperation source2 );

    virtual pBuffer read( const Interval& I );
    virtual pOperation source2() const { return source2_; }
    virtual IntervalType number_of_samples();

    virtual unsigned num_channels();
    virtual void set_channel(unsigned c);
    virtual unsigned get_channel() { return current_channel_; }

private:
    pOperation source2_;
    unsigned current_channel_;
};


class OperationSuperpositionChannels: public Operation, public boost::noncopyable
{
public:
    OperationSuperpositionChannels( pOperation source );

    virtual pBuffer read( const Interval& I );

    virtual unsigned num_channels() { return 1; }
    virtual void set_channel(unsigned c);
    virtual unsigned get_channel() { return 0; }

    virtual Signal::Intervals affected_samples();
};


} // namespace Signal

#endif // SIGNALOPERATIONBASIC_H
