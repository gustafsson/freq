#include "openurl.h"
#include "log.h"

//#define LOG_CALLS
#define LOG_CALLS if(0)

OpenUrl::OpenUrl(QQuickItem *parent) :
    QQuickItem(parent)
{
    LOG_CALLS Log("OpenUrl::OpenUrl");
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

void OpenUrl::
        openUrl(QUrl url)
{
    LOG_CALLS Log("OpenUrl::openUrl %s") % url.toString ().toStdString ();
}
