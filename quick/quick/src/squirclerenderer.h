#ifndef SQUIRCLERENDERER_H
#define SQUIRCLERENDERER_H

#include <QObject>
#include <QOpenGLShaderProgram>
#include "tools/rendermodel.h"
#include "tools/renderview.h"
#include "timer.h"

/**
 * @brief The SquircleRenderer is created when there is an OpenGL context and performs rendering.
 * The SquircleRenderer is destroyed before the OpenGL context is destroyed.
 */
class SquircleRenderer : public QObject {
    Q_OBJECT
public:
    SquircleRenderer(Tools::RenderModel* render_model);
    ~SquircleRenderer();

    void setT(qreal t) { m_t = t; }
    void setViewport(const QRectF &rect, double window_height, double ratio);

    Tools::RenderView* renderView() { return &render_view; }

signals:
    void redrawSignal();

public slots:
    void paint2();
    void paint();

private:
    Tools::RenderView render_view;

    QRect m_viewport;
    int m_window_height;
    qreal m_t, prevL=0;
    QOpenGLShaderProgram *m_program;
    Timer prevFrame;
};

#endif // SQUIRCLERENDERER_H
