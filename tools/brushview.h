#ifndef BRUSHVIEW_H
#define BRUSHVIEW_H

#include "brushmodel.h"
#include "heightmap/reference.h"

#include <QObject>

namespace Tools {

class BrushView: public QObject
{
    Q_OBJECT
public:
    BrushView(BrushModel* model);
    ~BrushView();

    bool enabled;
    Gauss gauss;

public slots:
    /// Connected in BrushController
    virtual void draw();

private:
    friend class BrushController;
    BrushModel* model_;

    void drawCircle();
};

} // namespace Tools

#endif // BRUSHVIEW_H
