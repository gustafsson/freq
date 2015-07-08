#ifndef SELECTIONRENDERER_H
#define SELECTIONRENDERER_H

#include <QObject>
#include "squirclerenderer.h"
#include "tvector.h"

class SelectionRenderer : public QObject
{
    Q_OBJECT
public:
    explicit SelectionRenderer(SquircleRenderer* parent);
    ~SelectionRenderer();

    void setSelection(double t1, double f1, double t2, double f2);
    void setSelection(Signal::Intervals);

    void setRgba(float r, float g, float b, float a);

signals:

public slots:
    void painting();

private:
    Tools::RenderModel* model = 0;
    float t1=0, f1=0, t2=0, f2=0;
    Signal::Intervals I;
    float r=0.0, g=0.0, b=0.0, a=0.5;
    QOpenGLShaderProgram *m_program = 0;

    void paint(float t1, float t2, float s1, float s2);
};

#endif // SELECTIONRENDERER_H
