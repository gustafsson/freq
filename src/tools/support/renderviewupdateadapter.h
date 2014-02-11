#ifndef TOOLS_SUPPORT_RENDERVIEWUPDATER_H
#define TOOLS_SUPPORT_RENDERVIEWUPDATER_H

#include "renderoperation.h"

#include <QObject>

namespace Tools {
class RenderView;

namespace Support {

/**
 * @brief The RenderViewUpdateAdapter class should translate the
 * Support::RenderOperationDesc::RenderTarget interface to Qt signals/slots
 * that match RenderView.
 *
 * It should not rely on a valid instance of RenderView
 *
 * It is up to the caller to connect the signals with a RenderView to forward information about processed data
 */
class RenderViewUpdateAdapter: public QObject, public Support::RenderOperationDesc::RenderTarget
{
    Q_OBJECT
public:
    RenderViewUpdateAdapter();

    // overloaded from Support::RenderOperationDesc::RenderTarget
    void refreshSamples(const Signal::Intervals& I);
    void processedData(const Signal::Interval& input, const Signal::Interval& output);

signals:
    void redraw();
    void setLastUpdateSize( Signal::UnsignedIntervalType length );

public:
    static void test();
};


class RenderViewUpdateAdapterMock: public QObject {
    Q_OBJECT
public:
    int redraw_count = 0;
    int setLastUpdateSize_count = 0;

public slots:
    void redraw();
    void setLastUpdateSize( Signal::UnsignedIntervalType );
};

} // namespace Support
} // namespace Tools

#endif // TOOLS_SUPPORT_RENDERVIEWUPDATEADAPTER_H
