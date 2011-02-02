#ifndef PLAYBACKMARKERSVIEW_H
#define PLAYBACKMARKERSVIEW_H

#include "playbackmarkersmodel.h"

#include <QObject>

namespace Sawe { class Project; }

namespace Tools {

class PlaybackMarkersView: public QObject
{
    Q_OBJECT
public:
    PlaybackMarkersView(PlaybackMarkersModel* model, Sawe::Project* project);

    PlaybackMarkersModel* model() { return model_; }

    bool enabled;
    void setHighlightMarker( float t, bool is_adding_marker );

public slots:
    /// Connected in EllipseController
    virtual void draw();

private:
    void drawMarkers();

    Sawe::Project* project_;
    PlaybackMarkersModel* model_;
    bool is_adding_marker_;
    float highlighted_marker_;
};

} // namespace Tools

#endif // PLAYBACKMARKERSVIEW_H
