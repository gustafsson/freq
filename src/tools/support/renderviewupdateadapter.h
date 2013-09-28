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
    // overloaded from Support::RenderOperationDesc::RenderTarget
    void refreshSamples(const Signal::Intervals& I);
    void processedData(const Signal::Interval& input, const Signal::Interval& output);

signals:
    void userinput_update();
    void setLastUpdateSize( Signal::IntervalType length );

public:
    static void test();
};


class RenderViewUpdateAdapterMock: public QObject {
    Q_OBJECT
public:
    int userinput_update_count = 0;
    int setLastUpdateSize_count = 0;

public slots:
    void userinput_update();
    void setLastUpdateSize( Signal::IntervalType );
};

} // namespace Support
} // namespace Tools

#endif // TOOLS_SUPPORT_RENDERVIEWUPDATEADAPTER_H
