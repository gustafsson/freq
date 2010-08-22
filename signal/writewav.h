#ifndef SIGNALWRITEWAV_H
#define SIGNALWRITEWAV_H

#include "signal/worker.h"
#include "signal/samplesintervaldescriptor.h"
#include "signal/sinksource.h"
#include <vector>

namespace Signal {

class WriteWav: public Sink
{
public:
    WriteWav( std::string filename );
    ~WriteWav();

    // Overloaded from Sink
    virtual void put( pBuffer b, pSource ) { put (b); }
    virtual void reset();
    virtual bool isFinished();
    virtual void onFinished();
    virtual SamplesIntervalDescriptor expected_samples() { return _data.expected_samples(); }
    virtual void add_expected_samples( const SamplesIntervalDescriptor& s ) { _data.add_expected_samples( s ); }

    static void writeToDisk(std::string filename, pBuffer b);

    void put( pBuffer );
private:
    SinkSource _data;
    std::string _filename;

    void writeToDisk();
};

} // namespace Signal

#endif // SIGNALWRITEWAV_H
