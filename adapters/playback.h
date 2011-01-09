#ifndef ADAPTERS_PLAYBACK_H
#define ADAPTERS_PLAYBACK_H

#include "signal/sinksource.h"

#include <vector>
#include <time.h>
#include <portaudiocpp/PortAudioCpp.hxx>
#include <boost/scoped_ptr.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

namespace Adapters {

class Playback: public Signal::Sink
{
public:
    Playback( int outputDevice/* = -1 */);
    ~Playback();

    // Overloaded from Sink
    virtual void put( Signal::pBuffer b, Signal::pOperation ) { put (b); }
    virtual bool isFinished();
    virtual Signal::Intervals fetch_invalid_samples() { return _data.fetch_invalid_samples(); }
    virtual void invalidate_samples( const Signal::Intervals& s ) { _data.invalidate_samples( s ); }

    void reset();
    void onFinished();

    static void list_devices();

    void put( Signal::pBuffer );
    unsigned    playback_itr();
    float       time();
    float       outputLatency();
    unsigned    output_device() { return _output_device; }
    bool        isStopped();
    bool        hasReachedEnd();
    bool        isUnderfed();
    float       sample_rate() { return _data.sample_rate(); }

    void        restart_playback();
private:
    Signal::SinkSource _data;
    boost::posix_time::ptime
            _first_timestamp,
            _last_timestamp,
            _startPlay_timestamp;
	unsigned _first_buffer_size;
    float _max_found,
          _min_found;

    void normalize( float* p, unsigned N );

    int readBuffer(const void * /*inputBuffer*/,
                     void *outputBuffer,
                     unsigned long framesPerBuffer,
                     const PaStreamCallbackTimeInfo *timeInfo,
                     PaStreamCallbackFlags statusFlags);

    portaudio::AutoSystem _autoSys;
    boost::scoped_ptr<portaudio::MemFunCallbackStream<Playback> > streamPlayback;

    long unsigned _playback_itr;
    int _output_device;
};

} // namespace Adapters

#endif // ADAPTERS_PLAYBACK_H
