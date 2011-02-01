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
    typedef float MarkerType;
    typedef std::set<MarkerType> Markers;
    Markers&            markers();
    void                removeMarker( Markers::iterator itr );
    void                addMarker( MarkerType pos );
    Markers::iterator   currentMarker();
    void                setCurrentMaker( Markers::iterator itr );

    /**
      Could return markers().end(), better check for that.
      */
    Markers::iterator   findMaker( MarkerType pos );
    Signal::Interval    currentInterval( float FS );

private:
    Markers::iterator current_marker_;
    Markers markers_;

    void                testIterator( Markers::iterator itr );
};

} // namespace Tools

#endif // PLAYBACKMARKERSMODEL_H
