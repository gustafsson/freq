#ifndef TOOLMAINLOOP_H
#define TOOLMAINLOOP_H

#include <QObject>

namespace Tools
{

class ToolMainLoop: public QObject
{
    Q_OBJECT
public:
    virtual ~ToolMainLoop() {}

};

} // namespace Tools

#endif // TOOLMAINLOOP_H
