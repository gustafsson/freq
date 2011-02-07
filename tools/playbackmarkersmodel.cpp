#include "playbackmarkersmodel.h"

#include <stdexcept>

namespace Tools {

PlaybackMarkersModel::
        PlaybackMarkersModel()
{
    addMarker( 0 );
}


PlaybackMarkersModel::Markers& PlaybackMarkersModel::
        markers()
{
    return markers_;
}


void PlaybackMarkersModel::
        removeMarker( Markers::iterator itr )
{
    if (itr == markers_.end() || itr == markers_.begin())
        return;

    testIterator( itr ); // will throw if itr is not valid

    if (current_marker_ == itr)
        current_marker_--;

    markers_.erase( itr );
}


void PlaybackMarkersModel::
        addMarker( MarkerType pos )
{
    std::pair<Markers::iterator, bool> value = markers_.insert( pos );
    current_marker_ = value.first;
    if (current_marker_ != markers_.begin())
        current_marker_--;
    // if value.second is false the element did already exist.
    // silently don't add a duplicate
}


PlaybackMarkersModel::Markers::iterator PlaybackMarkersModel::
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


PlaybackMarkersModel::Markers::iterator PlaybackMarkersModel::
        findMaker( MarkerType pos )
{
    // The ideal algortihm here would be a binary search, but this method is far from time critical
    Markers::iterator r = markers_.begin();
    for (Markers::iterator i = markers_.begin(); i != markers_.end(); ++i)
    {
        if (*i < pos)
            r = i;
        else
        {
            if(*i-pos < pos - *r)
                r = i;
            break;
        }
    }

    return r;
}


Signal::Interval PlaybackMarkersModel::
        currentInterval(float FS)
{
    Signal::Interval I = Signal::Interval::Interval_ALL;
    if (current_marker_ != markers_.end())
    {
        Markers::iterator next_marker = current_marker_;
        next_marker++;
        I.first = *current_marker_ * FS;
        if (next_marker != markers_.end())
            I.last = *next_marker * FS;
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
