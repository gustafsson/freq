#ifndef PLAYBACKMARKERSMODEL_H
#define PLAYBACKMARKERSMODEL_H

#include <set>
#include "signal/intervals.h"

namespace Tools {

class PlaybackMarkersModel
{
public:
    PlaybackMarkersModel();

    // set is an ordered vector
    typedef std::set<Signal::IntervalType> Markers;
    Markers&            markers();
    void                removeMarker( Markers::iterator itr );
    void                addMarker( Signal::IntervalType pos );
    Markers::iterator   currentMarker();
    void                setCurrentMaker( Markers::iterator itr );
    Signal::Interval    currentInterval();

private:
    Markers::iterator current_marker_;
    Markers markers_;

    void                testIterator( Markers::iterator itr );
};

} // namespace Tools

#endif // PLAYBACKMARKERSMODEL_H
