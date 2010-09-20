#ifndef SIGNALWRITEWAV_H
#define SIGNALWRITEWAV_H

#include "signal/worker.h"
#include "signal/intervals.h"
#include "signal/sinksource.h"
#include <vector>

namespace Signal {

class WriteWav: public Sink
{
public:
    WriteWav( std::string filename );
    ~WriteWav();

    // Overloaded from Sink
    virtual void put( pBuffer b, pOperation ) { put (b); }
    virtual void reset();
    // virtual Intervals expected_samples() { return _data.invalid_samples(); }
    virtual void invalidate_samples( const Intervals& s ) { _data.invalidate_samples( s ); }
    static void writeToDisk(std::string filename, pBuffer b);

    void put( pBuffer );

    static pBuffer crop(pBuffer b);
private:
    SinkSource _data;
    std::string _filename;

    void writeToDisk();
};

} // namespace Signal

#endif // SIGNALWRITEWAV_H
