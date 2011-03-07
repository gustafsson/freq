#ifndef FANTRACKERVIEW_H
#define FANTRACKERVIEW_H

#include <QObject>

namespace Tools {

class FanTrackerView : public QObject
{
    Q_OBJECT
public:
    explicit FanTrackerView(QObject *parent = 0);

signals:

public slots:
    virtual void draw();

};

} // namespace Tools

#endif // FANTRACKERVIEW_H
