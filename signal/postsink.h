#ifndef SIGNALPOSTSINK_H
#define SIGNALPOSTSINK_H

#include "sink.h"
#include <vector>
#ifndef SAWE_NO_MUTEX
#include <QMutex>
#endif

namespace Signal {

/**
  PostSink directs reads through a filter and lets sources chain-read through
  that filter if they promise not to change the outputs betweeen eachother.

  If they however will change their outputs PostSink will cache the read in a
  Buffer and let each source read from the same Buffer.
  */
class PostSink: public Sink
{
public:    
    /**
      For each Operation in sinks(), sets up a source and calls read(I). For
      performance reasons, different Operation's in sinks() may be chained into
      eachother rather than coupled to this->filter() directly. This only
      happens if they with affected_samples() promise to not change the data.

      this->filter() is coupled to this->source().

      @see Operation::affected_samples()
      */
    virtual Signal::pBuffer read( const Signal::Interval& I );


    /// @overload Operation::source()
    virtual pOperation source() { return Sink::source(); }


    /// @overload Operation::source(pOperation)
    virtual void source(pOperation v);


    /**
      Merges affected_samples() from all Operation's in sinks().
      */
    virtual Intervals affected_samples();


    /**
      Merges fetch_invalid_samples() from all Operation's in sinks(). Recursive
      behaviour is prevented by clearing Operation::source in each Operation
      first. Also, this->source()->fetch_invalid_samples() is not called.
      */
    virtual Intervals fetch_invalid_samples();


    /**
      Postsinks are removed by some controller when appropriate.
      */
    virtual bool isFinished() { return false; }


    /**
      A PostSink is underfed if any of its sinks are underfed.
      */
    virtual bool isUnderfed();


    /**
      Will call invalidate_samples on all instances of Signal::Sink in sinks().
      */
    virtual void invalidate_samples( const Intervals& I );


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

    pOperation              _filter;
#ifndef SAWE_NO_MUTEX
    QMutex                  _sinks_lock;
#endif
    std::vector<pOperation> _sinks;
};

} // namespace Signal

#endif // SIGNALPOSTSINK_H
