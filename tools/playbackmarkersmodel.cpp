#include "playbackmarkersmodel.h"

#include <stdexcept>

namespace Tools {

PlaybackMarkersModel::
        PlaybackMarkersModel()
{
    current_marker_ = markers_.end();
}


Markers& PlaybackMarkersModel::
        markers()
{
    return markers_;
}


void PlaybackMarkersModel::
        removeMarker( Markers::iterator itr )
{
    testIterator( itr ); // will throw if itr is not valid

    if (current_marker_ == itr)
        current_marker_++;

    markers_.erase( itr );
}


void PlaybackMarkersModel::
        addMarker( Signal::IntervalType pos )
{
    std::pair<Markers::iterator, bool> value = markers_.insert( pos );
    current_marker_ = value.first;
    // if value.second is false the element did already exist.
    // silently don't add a duplicate
}


Markers::iterator PlaybackMarkersModel::
        currentMarker()
{
    return current_marker_;
}


void PlaybackMarkersModel::
        setCurrentMaker( Markers::iterator itr )
{
    testIterator( itr ); // will throw if itr is not valid

    current_marker_ = itr;
}


Signal::Interval PlaybackMarkersModel::
        currentInterval()
{
    Signal::Interval I = Signal::Interval::Interval_ALL;
    if (current_marker_ != markers_.end())
    {
        I.first = *current_marker_;
        if (current_marker_ + 1 != markers_.end())
            I.last = *(current_marker_ + 1);
    }

    return I;
}


void PlaybackMarkersModel::
        testIterator( Markers::iterator itr )
{
    for (Markers::iterator i = markers_.begin(); i != markers_.end(); ++i)
        if (i == itr)
            return;

    throw std::logic_error("PlaybackMarkersModel, iterator does not point to a valid marker");
}


} // namespace Tools
