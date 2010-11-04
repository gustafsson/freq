#ifndef BRUSHVIEW_H
#define BRUSHVIEW_H

#include <QObject>

namespace Tools {

class BrushModel;

class BrushView: public QObject
{
    Q_OBJECT
public:
    BrushView(BrushModel* model);
    ~BrushView();

public slots:
    /// Connected in SelectionController
    virtual void draw();

private:
    friend class BrushController;
    BrushModel* model;
};

} // namespace Tools

#endif // BRUSHVIEW_H
