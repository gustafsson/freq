#ifndef DISPLAYWIDGET_H
#define DISPLAYWIDGET_H

#ifdef _WIN32 // QGLWidget includes WinDef.h on windows
#define NOMINMAX
#endif
#include <QGLWidget>
#include "heightmap/renderer.h"
#include "sawe/mainplayback.h"
#include "tfr/cwtfilter.h"
#include "signal/postsink.h"
#include <boost/shared_ptr.hpp>
#include <TAni.h>
#include <queue>
#include <QMainWindow>
#include "sawe/project.h"

#include "mousecontrol.h"
#include "tools/selectionmodel.h"
#include "tools/rendermodel.h"

namespace Tools{ class RenderView; }

namespace Ui {

class DisplayWidget :
        public QWidget
//        public Signal::Sink //, /* sink is used as microphone callback */
//        public QTimer && // TODO tidy
{
    Q_OBJECT
public:
    DisplayWidget( Sawe::Project* project, Tools::RenderView* render_view, Tools::RenderModel* render_model );

    virtual ~DisplayWidget();
    int lastKey;
    
    enum Yscale {
        Yscale_Linear,
        Yscale_ExpLinear,
        Yscale_LogLinear,
        Yscale_LogExpLinear
    } yscale;
    floatAni orthoview;

	bool isRecordSource();
    void setWorkerSource( Signal::pOperation s = Signal::pOperation());

    std::string selection_filename;
    unsigned playback_device;
    //Heightmap::Renderer* renderer();
    Tfr::CwtFilter* getCwtFilterHead(); // todo remove

protected:
    void open_inverse_test(std::string soundfile="");

    virtual void mousePressEvent ( QMouseEvent * e );
    virtual void mouseReleaseEvent ( QMouseEvent * e );
    virtual void wheelEvent ( QWheelEvent *event );
    virtual void mouseMoveEvent ( QMouseEvent * e );
    //virtual void timeOut();
    //void timerEvent( QTimerEvent *te);
    
protected slots:
    //virtual void timeOutSlot();
    virtual void receiveCurrentSelection(int, bool);
    virtual void receiveFilterRemoval(int);
    
    virtual void receiveToggleSelection(bool);
    virtual void receiveToggleNavigation(bool);
    virtual void receiveToggleInfoTool(bool);


    virtual void receivePlaySound();
    virtual void receiveFollowPlayMarker( bool v );
    virtual void receiveAddSelection(bool);
    virtual void receiveAddClearSelection(bool);

    virtual void receiveCropSelection();
    virtual void receiveMoveSelection(bool);
    virtual void receiveMoveSelectionInTime(bool);
    virtual void receiveMatlabOperation(bool);
    virtual void receiveMatlabFilter(bool);
    virtual void receiveTonalizeFilter(bool);
    virtual void receiveReassignFilter(bool);
    virtual void receiveRecord(bool);

signals:
    void renderingParametersChanged();
    void operationsUpdated( Signal::pOperation s );
    void setSelectionActive(bool);
    void setNavigationActive(bool);
    void setInfoToolActive(bool);

private:
    friend class Heightmap::Renderer;
    friend class Tools::RenderView;

    Sawe::Project* project;

    Tools::RenderModel* _render_model;
    Tools::RenderView* _render_view;

    Signal::pOperation _matlabfilter;
    Signal::pOperation _matlaboperation;

    bool _follow_play_marker;
    
    QTimer *_timer;
    int _prevX, _prevY, _targetQ;
    bool _selectionActive, _navigationActive, _infoToolActive;

    void drawArrows();
    void drawColorFace();
    void drawWaveform( Signal::pOperation waveform );
    static void drawWaveform_chunk_directMode( Signal::pBuffer chunk);
    static void drawSpectrogram_borders_directMode( Heightmap::pRenderer renderer );
    template<typename RenderData> void draw_glList( boost::shared_ptr<RenderData> chunk, void (*renderFunction)( boost::shared_ptr<RenderData> ), bool force_redraw=false );
    
    
    GLint viewport[4];
    GLdouble modelMatrix[16];
    GLdouble projectionMatrix[16];
    
    MyVector v1, v2;
    MyVector selectionStart;
    bool selecting;

    MyVector sourceSelection[2]; // todo, used for tool move selection
    
    void setSelection(int i, bool enabled);
    void removeFilter(int i);
    
    void locatePlaybackMarker();
    
    bool insideCircle( float x1, float z1 );


//    MouseControl leftButton;
//    MouseControl rightButton;
//    MouseControl middleButton;
    MouseControl selectionButton;
    MouseControl infoToolButton;
    MouseControl moveButton;
    MouseControl rotateButton;
    MouseControl scaleButton;
};

} // namespace Ui

#endif // DISPLAYWIDGET_H

