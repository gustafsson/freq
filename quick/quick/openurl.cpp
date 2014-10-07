#include "openurl.h"
#include "log.h"
#include "flacfile.h"
#include "qtaudiofile.h"
#include <QDesktopServices>

//#define LOG_CALLS
#define LOG_CALLS if(0)

OpenUrl::OpenUrl(QQuickItem *parent) :
    QQuickItem(parent)
{
    LOG_CALLS Log("OpenUrl::OpenUrl");

#ifdef Q_OS_IOS
    // cleanup old files
    QFileInfo fi(url.toLocalFile ());
    for (auto info : fi.dir ().entryInfoList())
    {
        if (info == fi || !info.isFile ())
            continue;

        Log("openurl: removing old file %s") % info.fileName ().toStdString ();
        fi.dir ().remove (info.fileName ());
    }
#endif

    QDesktopServices::setUrlHandler ( "file", this, "urlRequest");
}


void OpenUrl::
        componentComplete()
{
    QQuickItem::componentComplete();

    connect(this, SIGNAL(openUrl(QUrl)), this, SLOT(openUrl(QUrl)));
    LOG_CALLS Log("OpenUrl::componentComplete");
}


void OpenUrl::
        setChain(Chain* c)
{
    LOG_CALLS Log("OpenUrl::setChain");
    chain_=c;
}


Signal::OperationDesc::ptr parseFile(QUrl url)
{
    Signal::OperationDesc::ptr d;
    try {
        d.reset(new FlacFile(url));
        if (d->extent().sample_rate.is_initialized ())
            return d;
    } catch(...) {}

    try {
        d.reset(new QtAudiofile(url));
        if (d->extent().sample_rate.is_initialized ())
            return d;
    } catch(...) {}

    return Signal::OperationDesc::ptr();
}


void OpenUrl::
        openUrl(QUrl url)
{
    LOG_CALLS Log("OpenUrl::openUrl %s") % url.toString ().toStdString ();

    // first see if this was a valid file
    Signal::OperationDesc::ptr desc = parseFile(url);
    if (!desc)
        return;

    // purge target
    chain_->chain ()->removeOperationsAt(chain_->target_marker ());

    chain_->chain ()->addOperationAt(desc, chain_->target_marker ());
}
