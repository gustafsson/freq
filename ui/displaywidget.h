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

namespace Ui {

class DisplayWidget :
        public QGLWidget
//        public Signal::Sink //, /* sink is used as microphone callback */
//        public QTimer && // TODO tidy
{
    Q_OBJECT
public:
    DisplayWidget( Sawe::Project* project, QWidget* parent, Tools::RenderModel* model );

    virtual ~DisplayWidget();
    int lastKey;
    
    enum Yscale {
        Yscale_Linear,
        Yscale_ExpLinear,
        Yscale_LogLinear,
        Yscale_LogExpLinear
    } yscale;
    floatAni orthoview;
    float xscale;

	bool isRecordSource();
    void setWorkerSource( Signal::pOperation s = Signal::pOperation());

    std::string selection_filename;
    unsigned playback_device;
    Heightmap::pCollection collection();
    Heightmap::pRenderer renderer();
    Tfr::CwtFilter* getCwtFilterHead(); // todo remove

/*    virtual void keyPressEvent( QKeyEvent *e );
    virtual void keyReleaseEvent ( QKeyEvent * e );
    */
protected:
    void open_inverse_test(std::string soundfile="");
    virtual void initializeGL();
    virtual void resizeGL( int width, int height );
    virtual void paintGL();
    void setupCamera();

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
    virtual void receiveTogglePiano(bool);

    virtual void receiveSetRainbowColors();
    virtual void receiveSetGrayscaleColors();
    virtual void receiveSetHeightlines( bool value );
    virtual void receiveSetYScale( int value );
    virtual void receiveSetTimeFrequencyResolution( int value );

    virtual void receivePlaySound();
    virtual void receiveFollowPlayMarker( bool v );
    virtual void receiveToggleHz(bool);
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

    virtual void receiveSetTransform_Cwt();
    virtual void receiveSetTransform_Stft();
    virtual void receiveSetTransform_Cwt_phase();
    virtual void receiveSetTransform_Cwt_reassign();
    virtual void receiveSetTransform_Cwt_ridge();

signals:
    void renderingParametersChanged();
    void operationsUpdated( Signal::pOperation s );
    void setSelectionActive(bool);
    void setNavigationActive(bool);
    void setInfoToolActive(bool);

private:
    friend class Heightmap::Renderer;

    Sawe::Project* project;

    Tools::RenderModel* _model;

    Signal::pOperation _matlabfilter;
    Signal::pOperation _matlaboperation;
    boost::scoped_ptr<TaskTimer> _work_timer;
    boost::scoped_ptr<TaskTimer> _render_timer;

    bool _follow_play_marker;

    struct ListCounter {
        GLuint displayList;
        enum Age {
            Age_JustCreated,
            Age_InUse,
            Age_ProposedForRemoval
        } age;
        //ListAge age;
    };
    std::map<void*, ListCounter> _chunkGlList;
    
    QTimer *_timer;
    float _px, _py, _pz,
		_rx, _ry, _rz,
                _prevLimit;
    int _prevX, _prevY, _targetQ;
    bool _selectionActive, _navigationActive, _infoToolActive;
    QMutex _invalidRangeMutex;
    Signal::Intervals _invalidRange;

    void drawArrows();
    void drawColorFace();
    void drawWaveform( Signal::pOperation waveform );
    static void drawWaveform_chunk_directMode( Signal::pBuffer chunk);
    static void drawSpectrogram_borders_directMode( Heightmap::pRenderer renderer );
    template<typename RenderData> void draw_glList( boost::shared_ptr<RenderData> chunk, void (*renderFunction)( boost::shared_ptr<RenderData> ), bool force_redraw=false );
    
    bool _enqueueGcDisplayList;
    void gcDisplayList();
    
    GLint viewport[4];
    GLdouble modelMatrix[16];
    GLdouble projectionMatrix[16];
    
    MyVector v1, v2;
    MyVector selectionStart;
    bool selecting;

    MyVector sourceSelection[2]; // todo, used for tool move selection
    
    void setSelection(int i, bool enabled);
    void removeFilter(int i);
    
    void drawWorking();
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

