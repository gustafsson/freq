#ifndef SIGNALPLAYBACK_H
#define SIGNALPLAYBACK_H

#include "signal-sinksource.h"
#include <vector>
#include <time.h>
#include <QMutex>
#include <portaudiocpp/PortAudioCpp.hxx>
#include <boost/scoped_ptr.hpp>

namespace Signal {

class Playback: public SinkSource
{
public:
    Playback( int outputDevice/* = -1 */);
    ~Playback();

    // Overloaded from Sink
    virtual void put( pBuffer );
    virtual void put( pBuffer b, pSource ) { put (b); }
    virtual void reset();
    virtual bool finished();

    static void list_devices();

    unsigned    playback_itr();
    float       time();
    float       outputLatency();
    unsigned    output_device() { return _output_device; }
    bool        isStopped();
    bool        isUnderfed();

private:
    clock_t _first_timestamp;
    clock_t _last_timestamp;

    int readBuffer(const void * /*inputBuffer*/,
                     void *outputBuffer,
                     unsigned long framesPerBuffer,
                     const PaStreamCallbackTimeInfo *timeInfo,
                     PaStreamCallbackFlags statusFlags);

    portaudio::AutoSystem _autoSys;
    boost::scoped_ptr<portaudio::MemFunCallbackStream<Playback> > streamPlayback;

    unsigned _playback_itr;
    int _output_device;
};

} // namespace Signal

#endif // SIGNALPLAYBACK_H
