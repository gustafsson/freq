#ifndef SIGNALPOSTSINK_H
#define SIGNALPOSTSINK_H

#include "sink.h"
#include <vector>
#ifndef SAWE_NO_MUTEX
#include <QMutex>
#endif

namespace Signal {

//TODO rewrite PostSink, it's, error prone and ugle, i.e broken. The optimization that it once tried to achieve is no longer needed.

/**
  PostSink directs reads through a filter and lets sinks chain-read from
  eachother, through that filter and from the source if the sinks promise not
  to change the outputs betweeen eachother.

  If they however will change their outputs PostSink will cache the read in a
  Buffer and let each source read from the same Buffer.
  */
class PostSink: public Sink
{
public:    
    PostSink();

    /**
      For each Operation in sinks(), sets up a source and calls read(I). For
      performance reasons, different Operation's in sinks() may be chained into
      eachother rather than coupled to this->filter() directly. This only
      happens if they with affected_samples() promise to not change the data.

      this->filter() is coupled to this->source().

      @see Operation::affected_samples()
      */
    virtual Signal::pBuffer read( const Signal::Interval& I );


    /// @see readDirect
    pOperation readDirectSource();


    /// @overload Operation::source()
    virtual pOperation source();


    /// @overload Operation::source(pOperation)
    virtual void source(pOperation v);


    /**
      Merges affected_samples() from all Operation's in sinks().
      */
    virtual Intervals affected_samples();


    /**
      Merges invalid_samples() from all Sinks's in sinks(). Recursive
      behaviour is prevented by clearing Operation::source in each Operation
      first. Also, this->source()->invalid_samples() is not called even if
      source() were a Sink.
      */
    virtual Intervals invalid_samples();


    /**
      Postsinks are removed by some controller when appropriate.
      */
    virtual bool deleteMe() { return false; }


    /**
      A PostSink is underfed if any of its sinks are underfed.
      */
    virtual bool isUnderfed();
    bool isUnderfedIfInvalid;


    /**
      Call source()::Operation::invalidate_samples() to invalidate all instances of Signal::Sink in sinks().
      */
    virtual void invalidate_samples( const Intervals& );

    virtual std::string toString();

    /// @see read()
    std::vector<pOperation> sinks();
    void                    sinks(std::vector<pOperation> v);


    /**
      this->read() redirects reads through this filter which in turn reads from
      this->source(). When the filter is changed this->invalidate_samples( I )
      is called on all sinks() for those 'I' that might be changed.
      */
    pOperation              filter();
    void                    filter(pOperation f);

private:
    void                    update_source();
    void                    gcSinks();
    Signal::pBuffer         readDirect( const Signal::Interval& I );
    Signal::pBuffer         readSimple( const Signal::Interval& I );
    Signal::pBuffer         readActivePassive( const Signal::Interval& I );

    pOperation              _filter;
#ifndef SAWE_NO_MUTEX
    QMutex                  _sinks_lock;
#endif
    std::vector<pOperation> _sinks;
};

} // namespace Signal

#endif // SIGNALPOSTSINK_H
