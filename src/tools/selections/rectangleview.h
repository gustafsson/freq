#ifndef RECTANGLEVIEW_H
#define RECTANGLEVIEW_H

#include <QObject>

namespace Signal
{
    class Worker;
}

namespace Tools { namespace Selections
{
class RectangleModel;

class RectangleView: public QObject
{
    Q_OBJECT
public:
    RectangleView(RectangleModel* model, Signal::Worker* worker);
    ~RectangleView();

    bool enabled;
    bool visible;

    RectangleModel* model() { return model_; }

public slots:
    /// Connected in RectangleController
    virtual void draw();

private:
    void drawSelectionRectangle();
    void drawSelectionRectangle2();

    RectangleModel* model_;
    Signal::Worker* worker_;
};

}} // namespace Tools::Selections

#endif // RECTANGLEVIEW_H
