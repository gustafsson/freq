#ifndef GRAPHCONTROLLER_H
#define GRAPHCONTROLLER_H

#include <QObject>

namespace Signal
{
    class Worker;
}

namespace Tools
{

class RenderView;

class GraphController: public QObject
{
    Q_OBJECT
public:
    GraphController( RenderView* render_view );

    ~GraphController();

private slots:
    void redraw_operation_tree();

private:
    void setupGui();

    RenderView* render_view_;
    Signal::Worker* worker_;
};

} // namespace Tools

#endif // GRAPHCONTROLLER_H
