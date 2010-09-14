#ifndef SIGNALPOSTSINK_H
#define SIGNALPOSTSINK_H

#include "signal/operation.h"
#include <vector>
#include <QMutex>

namespace Signal {

/**
  PostSink directs reads through a filter and lets sources chain-read through
  that filter if they promise not to change the outputs betweeen eachother.

  If they however will change their outputs PostSink will cache the read in a
  Buffer and let each source read from the same Buffer.
  */
class PostSink: public Operation
{
public:
    PostSink(pOperation s=pOperation()):Operation(s) {}
    virtual Signal::pBuffer read( const Signal::Interval& I );

    virtual Intervals affected_samples();
    virtual Intervals invalid_samples();
    virtual void invalidate_samples( const Intervals& s );

    std::vector<pOperation> sinks();
    void                    sinks(std::vector<pOperation> v);
    pOperation              filter();
    void                    filter(pOperation f);

private:

    pOperation              _filter;
    QMutex                  _sinks_lock;
    std::vector<pOperation> _sinks;
};

} // namespace Signal

#endif // SIGNALPOSTSINK_H
