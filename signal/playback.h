#ifndef SIGNALPLAYBACK_H
#define SIGNALPLAYBACK_H

#include "signal/sinksource.h"
#include <vector>
#include <time.h>
#include <QMutex>
#include <portaudiocpp/PortAudioCpp.hxx>
#include <boost/scoped_ptr.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

namespace Signal {

class Playback: public Sink
{
public:
    Playback( int outputDevice/* = -1 */);
    ~Playback();

    // Overloaded from Sink
    virtual void put( pBuffer b, pOperation ) { put (b); }
    virtual bool isFinished();
    virtual Intervals invalid_samples() { return _data.invalid_samples(); }
    virtual void invalidate_samples( const Intervals& s ) { _data.invalidate_samples( s ); }

    void reset();
    void onFinished();

    static void list_devices();

    void put( pBuffer );
    unsigned    playback_itr();
    float       time();
    float       outputLatency();
    unsigned    output_device() { return _output_device; }
    bool        isStopped();
    bool        isUnderfed();
    unsigned    sample_rate() { return _data.sample_rate(); }
private:
    SinkSource _data;
    boost::posix_time::ptime
            _first_timestamp,
            _last_timestamp,
            _startPlay_timestamp;
	unsigned _first_buffer_size;

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
