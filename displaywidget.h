#ifndef DISPLAYWIDGET_H
#define DISPLAYWIDGET_H

#include <QGLWidget>
#include "spectrogram.h"
#include <boost/shared_ptr.hpp>

class DisplayWidget : public QGLWidget
{
public:
    DisplayWidget( boost::shared_ptr<Spectrogram> spectrogram, int timerInterval=0 );
    ~DisplayWidget();
  static int lastKey;

protected:
  virtual void initializeGL();
  virtual void resizeGL( int width, int height );
  virtual void paintGL();

  virtual void mousePressEvent ( QMouseEvent * e );
  virtual void mouseMoveEvent ( QMouseEvent * e );
  virtual void timeOut();

protected slots:
  virtual void timeOutSlot();

private:
  boost::shared_ptr<Spectrogram> _spectrogram;
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
        _qx, _qy, _qz;
  int _prevX, _prevY;

  void drawArrows();
  void drawColorFace();
  void drawWaveform( pWaveform waveform );
  static void drawWaveform_chunk_directMode( pWaveform_chunk chunk);
  void drawSpectrogram();
  static void drawSpectrogram_block_directMode( Spectrogram::pBlock block );
  void drawSpectrogram_block_vertexArray( Spectrogram::pBlock block );
  void drawSpectrogram_block_cudaGlInterop( Spectrogram::pBlock block );
  void drawSpectrogram_block_cgShader( Spectrogram::pBlock block );

  void renderSpectrogramRef( Spectrogram::Reference ref );
  bool renderChildrenSpectrogramRef( Spectrogram::Reference ref );
  void renderParentSpectrogramRef( Spectrogram::Reference ref );

  template<typename RenderData> void draw_glList( boost::shared_ptr<RenderData> chunk, void (*renderFunction)( boost::shared_ptr<RenderData> ) );

  bool _enqueueGcDisplayList;
  void gcDisplayList();
};

#endif // DISPLAYWIDGET_H

