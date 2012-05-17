#ifndef ADAPTERS_PLAYBACK_H
#define ADAPTERS_PLAYBACK_H

#include "signal/sinksourcechannels.h"

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
    virtual bool deleteMe();
    virtual void invalidate_samples( const Signal::Intervals& s );
    virtual unsigned num_channels();
    virtual void set_channel(unsigned c);
    virtual Signal::Intervals invalid_samples();

    void stop();
    void reset();
    void onFinished();

    static void list_devices();
    struct DeviceInfo
    {
        int index;
        std::string name;
        int inputChannels, outputChannels;
        std::string name2;
        bool isDefaultIn, isDefaultOut;
    };

    static std::list<DeviceInfo> get_devices();

    void put( Signal::pBuffer );
    unsigned    playback_itr();
    float       time();
    float       outputLatency();
    unsigned    output_device() { return _output_device; }
    bool        isStopped();
    bool        hasReachedEnd();
    bool        isUnderfed();
    bool        isPaused();
    void        pausePlayback(bool pause);
    float       sample_rate() { return _data.sample_rate(); }

    void        restart_playback();
private:
    Signal::SinkSourceChannels _data;
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

    Signal::IntervalType _playback_itr;
    int _output_device;
    bool _is_interleaved;
};

} // namespace Adapters

#endif // ADAPTERS_PLAYBACK_H
