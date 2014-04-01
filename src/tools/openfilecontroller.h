#ifndef TOOLS_OPENFILECONTROLLER_H
#define TOOLS_OPENFILECONTROLLER_H

#include "signal/operation.h"

#include <QObject>
#include <QPointer>

namespace Tools {

/**
 * @brief The OpenfileController class should provide a generic interface for
 * opening files.
 */
class OpenfileController : public QObject
{
    Q_OBJECT
public:
    explicit OpenfileController(QObject *parent = 0);
    static OpenfileController* instance();

    typedef QList<std::pair<QString,QString> > Patterns;

    class OpenfileInterface : public QObject {
    public:
        typedef Patterns Patterns;

        virtual Patterns patterns() = 0;
        virtual Signal::OperationDesc::ptr open(QString url) = 0;
    };

    QList<QPointer<OpenfileInterface> > get_openers() { return file_openers; }
    void registerOpener(QPointer<OpenfileInterface> file_opener);
    Patterns patterns();

    Signal::OperationDesc::ptr open(QString url);

signals:

public slots:

private:
    QList<QPointer<OpenfileInterface> > file_openers;

public:
    static void test();
};

} // namespace Tools

#endif // TOOLS_OPENFILECONTROLLER_H
