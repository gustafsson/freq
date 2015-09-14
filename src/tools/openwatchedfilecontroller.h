#ifndef TOOLS_OPENWATCHEDFILECONTROLLER_H
#define TOOLS_OPENWATCHEDFILECONTROLLER_H

#include "openfilecontroller.h"

namespace Tools {

/**
 * @brief The OpenWatchedFileController class should reopen a file when it is modified.
 *
 * There is a delay to reloading the file to avoid interference if the file is
 * modified repeatedly (as in, being continuously written to). OpenfileController
 * is not asked to reload the file until it has been left alone for 'delay_ms'.
 *
 * Uses OpenfileController
 */
class OpenWatchedFileController : public QObject
{
    Q_OBJECT
public:
    explicit OpenWatchedFileController(QPointer<OpenfileController> openfilecontroller, int delay_ms=250);

    Signal::OperationDesc::ptr openWatched(QString url);

signals:

public slots:

private:
    QPointer<OpenfileController> openfilecontroller_;
    int delay_ms_;

public:
    static void test();
};


class FileChangedBase: public QObject
{
    Q_OBJECT
public slots:
    virtual void fileChanged ( const QString & path) = 0;
    virtual void delayedFileChanged () = 0;
};

} // namespace Tools

#endif // TOOLS_OPENWATCHEDFILECONTROLLER_H
