#ifndef DISPLAYWIDGET_H
#define DISPLAYWIDGET_H

#ifdef _WIN32 // QGLWidget includes WinDef.h on windows
#define NOMINMAX
#endif
#include <QGLWidget>
#include "heightmap-renderer.h"
#include "sawe-mainplayback.h"
#include "signal-filteroperation.h"
#include "signal-postsink.h"
#include <boost/shared_ptr.hpp>
#include <TAni.h>
#include <queue>
#include <QMainWindow>

class MouseControl
{
private:
    float lastx;
    float lasty;
    bool down;
    unsigned int hold;
    
public:
    MouseControl(): down( false ), hold( 0 ) {}
    
    float deltaX( float x );
    float deltaY( float y );
    
    bool worldPos(GLdouble &ox, GLdouble &oy);
    static bool worldPos(GLdouble x, GLdouble y, GLdouble &ox, GLdouble &oy);
    /**
      worldPos projects space coordinates onto the xz-plane. spacePos simple returns the space pos.
      */
    bool spacePos(GLdouble &out_x, GLdouble &out_y);
    static bool spacePos(GLdouble in_x, GLdouble in_y, GLdouble &out_x, GLdouble &out_y);

    bool isDown(){return down;}
    bool isTouched();
    int getHold(){return hold;}
    
    void press( float x, float y );
    void update( float x, float y );
    void release();
    void touch(){hold = 0;}
    void untouch(){hold++;}
};

struct MyVector{
    float x, y, z;
};

class DisplayWidget :
        public QGLWidget,
        public Signal::Sink //, /* sink is used as microphone callback */
//        public QTimer
{
    Q_OBJECT
public:
    DisplayWidget( Signal::pWorker worker, Signal::pSink collection );
    virtual ~DisplayWidget();
    int lastKey;
    static DisplayWidget* gDisplayWidget;
    
    enum Yscale {
        Yscale_Linear,
        Yscale_ExpLinear,
        Yscale_LogLinear,
        Yscale_LogExpLinear
    } yscale;
    floatAni orthoview;
    float xscale;

	bool isRecordSource();
    void setWorkerSource( Signal::pSource s = Signal::pSource());
    void setTimeline( Signal::pSink timelineWidget );
    void setPosition( float time, float f );

    Signal::pWorker worker() { return _worker; }
    std::string selection_filename;
    unsigned playback_device;
    Heightmap::Collection* collection() { return _renderer->collection(); }
    Heightmap::pRenderer renderer() { return _renderer; }

    void drawSelection();

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
    virtual void receiveTogglePiano(bool);


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
    virtual void receiveRecord(bool);
signals:
    void operationsUpdated( Signal::pSource s );
    void filterChainUpdated( Tfr::pFilter f );
    void setSelectionActive(bool);
    void setNavigationActive(bool);
    
private:
    friend class Heightmap::Renderer;

    // overloaded from Signal::Sink
    virtual void put( Signal::pBuffer b, Signal::pSource );
    virtual void add_expected_samples( const Signal::SamplesIntervalDescriptor& s );

    Signal::FilterOperation* getFilterOperation();
    Signal::PostSink* getPostSink();

    Heightmap::pRenderer _renderer;
    Signal::pWorker _worker;
    Signal::pWorkerCallback _collectionCallback;
    Signal::pWorkerCallback _postsinkCallback;
    Signal::pSource _matlabfilter;
    Signal::pSource _matlaboperation;
    Signal::pSink _timeline;
    boost::scoped_ptr<TaskTimer> _work_timer;

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
		_qx, _qy, _qz,
                _prevLimit,
                _playbackMarker;
    int _prevX, _prevY, _targetQ;
    bool _selectionActive, _navigationActive;
    QMutex _invalidRangeMutex;
    Signal::SamplesIntervalDescriptor _invalidRange;

    void drawArrows();
    void drawColorFace();
    void drawWaveform( Signal::pSource waveform );
    static void drawWaveform_chunk_directMode( Signal::pBuffer chunk);
    static void drawSpectrogram_borders_directMode( Heightmap::pRenderer renderer );
    template<typename RenderData> void draw_glList( boost::shared_ptr<RenderData> chunk, void (*renderFunction)( boost::shared_ptr<RenderData> ), bool force_redraw=false );
    
    bool _enqueueGcDisplayList;
    void gcDisplayList();
    
    GLint viewport[4];
    GLdouble modelMatrix[16];
    GLdouble projectionMatrix[16];
    
    MyVector v1, v2;
    MyVector selection[2], selectionStart;
    bool selecting;

    MyVector sourceSelection[2];
    
    void setSelection(int i, bool enabled);
    void removeFilter(int i);
    
    void drawWorking();
    void locatePlaybackMarker();
    void drawPlaybackMarker();
    void drawSelectionCircle();
    void drawSelectionCircle2();
    void drawSelectionSquare();
    
    bool insideCircle( float x1, float z1 );
    

    MouseControl leftButton;
    MouseControl rightButton;
    MouseControl middleButton;
    MouseControl selectionButton;
    MouseControl moveButton;
    MouseControl rotateButton;
    MouseControl scaleButton;
};


#endif // DISPLAYWIDGET_H

