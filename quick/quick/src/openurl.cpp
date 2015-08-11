#include "openurl.h"
#include "log.h"
#include "flacfile.h"
#include "qtaudiofile.h"
#include "signal/processing/workers.h"
#include <QDesktopServices>

//#define LOG_CALLS
#define LOG_CALLS if(0)

OpenUrl::OpenUrl(QQuickItem *parent) :
    QQuickItem(parent)
{
    LOG_CALLS Log("OpenUrl::OpenUrl");

    QDesktopServices::setUrlHandler ( "file", this, "onOpenUrl");
}


void OpenUrl::
        componentComplete()
{
    QQuickItem::componentComplete();

    connect(this, SIGNAL(openUrl(QUrl)), this, SLOT(onOpenUrl(QUrl)));
    LOG_CALLS Log("OpenUrl::componentComplete");
}


void OpenUrl::
        setChain(Chain* c)
{
    LOG_CALLS Log("OpenUrl::setChain");
    if (c==chain_)
        return;
    chain_=c;
    emit chainChanged();
}


Signal::OperationDesc::ptr parseFile(QUrl url)
{
    Signal::OperationDesc::ptr d;
    try {
#ifndef TARGET_IPHONE_SIMULATOR
        d.reset(new FlacFile(url));
        if (d->extent().sample_rate.is_initialized ())
            return d;
#endif
    } catch(...) {}

    try {
        d.reset(new QtAudiofile(url));
        if (d->extent().sample_rate.is_initialized ())
            return d;
    } catch(...) {}

    return Signal::OperationDesc::ptr();
}


void OpenUrl::
        onOpenUrl(QUrl url)
{
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

    LOG_CALLS Log("OpenUrl::onOpenUrl %s") % url.toString ().toStdString ();

    // first see if this was a valid file
    Signal::OperationDesc::ptr desc = parseFile(url);
    if (!desc)
        return;

    // purge target
    chain_->chain ()->removeOperationsAt(chain_->target_marker ());

    chain_->chain ()->addOperationAt(desc, chain_->target_marker ());
    chain_->setTitle (url.fileName ());
    chain_->chain ()->workers()->addComputingEngine(Signal::ComputingEngine::ptr(new Signal::DiscAccessThread));
}
