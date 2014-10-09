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
signals:

public slots:
    void painting();

private:
    Tools::RenderModel* model = 0;
    float t1=0, f1=0, t2=0, f2=0;
    QOpenGLShaderProgram *m_program = 0;
};

#endif // SELECTIONRENDERER_H
