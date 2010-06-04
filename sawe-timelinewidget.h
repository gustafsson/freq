#ifndef SAWETIMELINEWIDGET_H
#define SAWETIMELINEWIDGET_H

#include <QGLWidget>
#include "signal-sink.h"
#include "displaywidget.h"

namespace Sawe {

/**
  Transforms a pBuffer into a pChunk with CwtSingleton and saves the chunk in a file called
  sonicawe-x.csv, where x is a number between 1 and 9, or 0 if all the other 9 files already
  exists. The file is saved with the csv-format comma separated values, but values are
  actually separated by spaces. One row of the csv-file corresponds to one row of the chunk.
*/
class TimelineWidget:
        public QGLWidget,
        public Signal::Sink
{
    Q_OBJECT
public:
    TimelineWidget( Signal::pSink displayWidget );
    virtual ~TimelineWidget();

    // overloaded from Signal::Sink
    virtual void    put( Signal::pBuffer , Signal::pSource );
    virtual void    add_expected_samples( const Signal::SamplesIntervalDescriptor& );

protected:
    // overloaded from QGLWidget
    virtual void initializeGL();
    virtual void resizeGL( int width, int height );
    virtual void paintGL();

    // overloaded from QWidget
    virtual void mousePressEvent ( QMouseEvent * e );
    virtual void mouseReleaseEvent ( QMouseEvent * e );
    virtual void mouseMoveEvent ( QMouseEvent * e );
private:
    MouseControl moveButton;

    Signal::pSink _displayWidget;
    DisplayWidget * getDisplayWidget();
};

} // namespace Sawe

#endif // SAWECSV_H
