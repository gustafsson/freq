#ifndef SIGNALSOURCE_H
#define SIGNALSOURCE_H

#include "buffer.h"

namespace Signal {


/**
Signal::Source is the single most important class in the Signal namespace. It
declares an interface through which buffers can be read.
*/
class SaweDll SourceBase
{
public:
    /**
      Virtual housekeeping.
     */
    virtual ~SourceBase() {}


    /**
      read does not have to return a Buffer of the same size as I. But it has
      to include I.first. The caller of read must allow for read to return
      Buffers of arbitrary sizes.

      However, read shall never return a null pBuffer(). Either throw an
      logic_error() exception or create a new Buffer with zeros.
    */
    virtual pBuffer read( const Interval& I ) = 0;
    virtual float sample_rate() = 0;
    virtual IntervalType number_of_samples() = 0;
    virtual unsigned num_channels() = 0;
    virtual Interval getInterval() { return Interval(0, number_of_samples() ); }


    /**
      Creates a buffer of the exact interval 'I'. Takes data from 'read' and
      crops it if necessary.
      */
    pBuffer readFixedLength( const Interval& I );


    /**
      Length of signal, in seconds.
      */
    float length() { return number_of_samples() / sample_rate(); }
    std::string lengthLongFormat() { return lengthLongFormat(length()); }
    static std::string lengthLongFormat( float T );


protected:
    /**
      Used by 'readFixedLength' and others to create a buffer with zeros. The
      buffer will be over the exact interval 'I'.
      */
    pBuffer zeros( const Interval& I );


private:
    /**
      Calls 'read' and checks that it returns valid data. That is; a non-empty
      buffer containing I.first. Used by 'readFixedLength'.
      */
    pBuffer readChecked( const Interval& I );
};


} // namespace Signal

#endif // SIGNALSOURCE_H
