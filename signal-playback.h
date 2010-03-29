#ifndef SIGNALPLAYBACK_H
#define SIGNALPLAYBACK_H

#include "signal-sink.h"
#include <list>
#include <time.h>
#include <portaudiocpp/PortAudioCpp.hxx>
#include <boost/scoped_ptr.hpp>

namespace Signal {

class Playback: public Sink
{
public:
    Playback( /* int outputDevice = -1 */ );

    virtual void put( pBuffer );

    static void list_devices();
    unsigned playback_itr();

private:
    struct BufferSlot {
        pBuffer buffer;
        clock_t timestamp;
    };

    int readBuffer(const void */*inputBuffer*/,
                     void *outputBuffer,
                     unsigned long framesPerBuffer,
                     const PaStreamCallbackTimeInfo *timeInfo,
                     PaStreamCallbackFlags statusFlags);

    boost::scoped_ptr<portaudio::MemFunCallbackStream<Playback> > streamPlayback;

    std::vector<BufferSlot> _cache;
    unsigned _playback_itr;

    void nAccumulatedSamples();
};

} // namespace Signal

#endif // SIGNALPLAYBACK_H
