#ifndef TOOLS_OPENWATCHEDFILECONTROLLER_H
#define TOOLS_OPENWATCHEDFILECONTROLLER_H

#include "openfilecontroller.h"

namespace Tools {

/**
 * @brief The OpenWatchedFileController class should reopen a file when it is modified.
 *
 * Uses OpenfileController
 */
class OpenWatchedFileController : public QObject
{
    Q_OBJECT
public:
    explicit OpenWatchedFileController(QPointer<OpenfileController> openfilecontroller);

    Signal::OperationDesc::ptr openWatched(QString url);

signals:

public slots:

private:
    QPointer<OpenfileController> openfilecontroller_;

public:
    static void test();
};


class FileChangedBase: public QObject
{
    Q_OBJECT
public slots:
    virtual void fileChanged ( const QString & path) = 0;
};

} // namespace Tools

#endif // TOOLS_OPENWATCHEDFILECONTROLLER_H
