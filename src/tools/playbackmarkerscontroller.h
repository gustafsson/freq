#ifndef PLAYBACKMARKERSCONTROLLER_H
#define PLAYBACKMARKERSCONTROLLER_H

#include "playbackmarkersview.h"

#include <QWidget>

namespace Sawe {
        class Project;
}

namespace Tools {

class RenderView;

class PlaybackMarkersController: public QWidget
{
    Q_OBJECT
public:
    PlaybackMarkersController(PlaybackMarkersView* view, RenderView* render_view, Sawe::Project* project);

signals:
    void enabledChanged(bool enabled);

public slots:
    void enableMarkerTool(bool active);
    void receivePreviousMarker();
    void receiveNextMarker();

private:
    // Event handlers
    virtual void mousePressEvent ( QMouseEvent * e );
    virtual void mouseMoveEvent ( QMouseEvent * e );
    virtual void changeEvent(QEvent *);

    void setupGui();

    float vicinity_; // in pixels
    RenderView* render_view_;
    PlaybackMarkersView* view_;
    Sawe::Project* project_;
    PlaybackMarkersModel* model() { return view_->model(); }
};

} // namespace Tools

#endif // PLAYBACKMARKERSCONTROLLER_H
