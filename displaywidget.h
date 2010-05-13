#ifndef DISPLAYWIDGET_H
#define DISPLAYWIDGET_H

#include <QGLWidget>
#include "heightmap-renderer.h"
#include "sawe-mainplayback.h"
#include "signal-filteroperation.h"
#include <boost/shared_ptr.hpp>
#include <TAni.h>
#include <queue>

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

class DisplayWidget : public QGLWidget, public Signal::Sink /* sink is used as microphone callback */
{
    Q_OBJECT
public:
    DisplayWidget( Signal::pWorker worker, Signal::pSink collection, unsigned playback_device, std::string selection_filename, int timerInterval=0  );
    ~DisplayWidget();
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

    void setWorkerSource( Signal::pSource s = Signal::pSource());

    virtual void keyPressEvent( QKeyEvent *e );
    virtual void keyReleaseEvent ( QKeyEvent * e );
protected:
    void open_inverse_test(std::string soundfile="");
    virtual void initializeGL();
    virtual void resizeGL( int width, int height );
    virtual void paintGL();
    
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
    virtual void receiveToggleHz(bool);
    virtual void receiveAddSelection(bool);
    virtual void receiveAddClearSelection(bool);

    virtual void receiveCropSelection();
    virtual void receiveMoveSelection(bool);
    virtual void receiveMoveSelectionInTime(bool);
signals:
    void operationsUpdated( Signal::pSource s );
    void filterChainUpdated( Tfr::pFilter f );
    void setSelectionActive(bool);
    void setNavigationActive(bool);
    
private:
    friend class Heightmap::Renderer;

    virtual void put( Signal::pBuffer b);
//    virtual void put( Signal::pBuffer, Signal::Source* );
    Signal::FilterOperation* getFilterOperation();
    // bool _record_update;

    Heightmap::pRenderer _renderer;
    Signal::pWorker _worker;
    Signal::pWorkerCallback _collectionCallback;
    Signal::pWorkerCallback _playbackCallback;
    Signal::pWorkerCallback _diskwriterCallback;

    std::string _selection_filename;
    unsigned _playback_device;

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
		_renderRatio;
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
    
    void drawSelection();
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

