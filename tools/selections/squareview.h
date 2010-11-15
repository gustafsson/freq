#ifndef SQUAREVIEW_H
#define SQUAREVIEW_H

#include <QObject>
#include "signal/worker.h"

namespace Tools { namespace Selections
{

class SquareModel;

class SquareView: public QObject
{
    Q_OBJECT
public:
    SquareView(SquareModel* model, Signal::Worker* worker);
    ~SquareView();

    void drawSelectionRectangle();
    void drawSelectionRectangle2();

    bool enabled;

public slots:
    /// Connected in SquareController
    virtual void draw();

private:
    friend class SquareController;
    SquareModel* model_;
    Signal::Worker* worker_;
};

}} // namespace Tools::Selections

#endif // SQUAREVIEW_H
