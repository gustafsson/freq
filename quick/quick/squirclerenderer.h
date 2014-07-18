#ifndef SQUIRCLERENDERER_H
#define SQUIRCLERENDERER_H

#include <QObject>
#include <QOpenGLShaderProgram>
#include "tools/rendermodel.h"
#include "tools/renderview.h"

class SquircleRenderer : public QObject {
    Q_OBJECT
public:
    SquircleRenderer();
    ~SquircleRenderer();

    void setT(qreal t) { m_t = t; }
    void setViewportSize(const QSize &size) { m_viewportSize = size; }

public slots:
    void paint();

private:
    Tools::RenderModel render_model;
    Tools::RenderView render_view;

    QSize m_viewportSize;
    qreal m_t;
    QOpenGLShaderProgram *m_program;
};

#endif // SQUIRCLERENDERER_H
