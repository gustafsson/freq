#ifndef SAWEMAINPLAYBACK_H
#define SAWEMAINPLAYBACK_H

#include <boost/shared_ptr.hpp>
#include "signal-playback.h"
#include "signal-worker.h"

namespace Sawe {

class MainPlayback: public Signal::WorkerCallback
{
public:
    MainPlayback(int outputDevice, Signal::Worker* worker):WorkerCallback(worker), pb(outputDevice) {}

    /**
      Signal::Playback outputs audio data through port audio.
      */
    Signal::Playback pb;

private:
    virtual void reset() { pb.reset(); }
    virtual void put( Signal::pBuffer b ) { pb.put( b ); }
    virtual void put( Signal::pBuffer b, Signal::Source* s ) { pb.Sink::put (b, s); }
};
typedef boost::shared_ptr<MainPlayback> pMainPlayback;

} // namespace Sawe

#endif // SAWEMAINPLAYBACK_H
