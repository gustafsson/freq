#include "touchnavigation.h"
#include "tools/support/renderviewinfo.h"
#include "log.h"

using Tools::Support::RenderViewInfo;

TouchNavigation::TouchNavigation(QObject* parent, Tools::RenderModel* render_model)
    :
      QObject(parent),
      render_model(render_model)
{
}


void TouchNavigation::
        mouseMove(qreal x1, qreal y1, bool p1)
{
    touch(x1,y1,p1,0,0,0,0,0,0);
}


void TouchNavigation::
        touch(qreal x1, qreal y1, bool press1, qreal x2, qreal y2, bool press2, qreal x3, qreal y3, bool press3)
{
    if (press3)
        press1 = press2 = press3 = false;

    QPointF point1(x1,y1), point2(x2,y2);
    Heightmap::Position hpos1, hpos2;

    auto c = *render_model->camera.read();
    vectord& q = c.q;
    vectord& r = c.r;

    if (press1 && !press2)
        hpos1 = RenderViewInfo(render_model).getPlanePos (point1);
    else if (press1 && press2)
        hpos1 = RenderViewInfo(render_model).getPlanePos ((point1+point2)/2);

    if (press1 && !press2 && prevPress1 && !prevPress2)
    {
        // pan is based on the idea that a touched position should still point
        // at the same position when its moving, hence the delta here is not how much the point
        // has moved since the last step but how wrong the current position is compared to the
        // starting position
        double dtime1 = hpos1.time - hstart1.time;
        double dscale1 = hpos1.scale - hstart1.scale;
        q[0] -= dtime1/2;
        q[2] -= dscale1/2;

        Log("(%g, %g)->(%g, %g): q[0] %g. q[2] %g") % x1 % y1 % hpos1.time % hpos1.scale % q[0] % q[2];
    }
    else if (press1 && press2 && !press3 && prevPress1 && prevPress2)
    {
        if (Undefined == mode)
        {
            qreal sx1 = point1.x () - start1.x ();
            qreal sy1 = point1.y () - start1.y ();
            qreal sx2 = point2.x () - start2.x ();
            qreal sy2 = point2.y () - start2.y ();
            qreal ax1 = std::abs (sx1), ax2 = std::abs (sx2);
            qreal ay1 = std::abs (sy1), ay2 = std::abs (sy2);

            int threshold = 15;
            // If one of the points have moved above a threshold, Scale
            if (ax1 > threshold || ax2 > threshold || ay1 > threshold || ay2 > threshold)
                mode = Scale;

            // If both are moved and they have the same sign, Rotate
            if ((sx1 > 0) == (sx2 > 0))
                if ((ax1 > threshold && ax2 > threshold/2) || (ax1 > threshold/2 && ax2 > threshold))
                    mode = Rotate;
            if ((sy1 > 0) == (sy2 > 0))
                if ((ay1 > threshold && ay2 > threshold/2) || (ay1 > threshold/2 && ay2 > threshold))
                    mode = Rotate;
        }

        auto clamp =
                [](qreal min, qreal max, qreal v) {
                    return std::min(max, std::max(min, v));
                };

        switch (mode)
        {
        case Rotate:
        {
            qreal dx1 = point1.x () - prev1.x ();
            qreal dy1 = point1.y () - prev1.y ();
            qreal dx2 = point2.x () - prev2.x ();
            qreal dy2 = point2.y () - prev2.y ();

            r[1] += (dx1 + dx2)/6;
            r[0] += (dy1 + dy2)/6;

            Log("(%g, %g)  (%g, %g): r[0] %g. r[1] %g")
                    % x1 % y1 % x2 % y2 % r[0] % r[1];
            break;
        }
        case Scale:
        {
            QTransform T; T.rotate (c.effective_ry ());

            QPointF rpoint1 = T.map (point1);
            QPointF rpoint2 = T.map (point2);
            QPointF rprev1 = T.map (prev1);
            QPointF rprev2 = T.map (prev2);

            qreal dx1 = rpoint1.x () - rprev1.x ();
            qreal dy1 = rpoint1.y () - rprev1.y ();
            qreal dx2 = rpoint2.x () - rprev2.x ();
            qreal dy2 = rpoint2.y () - rprev2.y ();

            if (rpoint1.x () > rpoint2.x ()) std::swap(dx1,dx2);
            if (rpoint1.y () > rpoint2.y ()) std::swap(dy1,dy2);

            c.xscale *= clamp(0.95, 1.05, 1 - (dx1-dx2)/100);
            c.zscale *= clamp(0.95, 1.05, 1 - (dy1-dy2)/100);

            // Also pan
            double dtime1 = hpos1.time - hstart1.time;
            double dscale1 = hpos1.scale - hstart1.scale;
            q[0] -= dtime1/2;
            q[2] -= dscale1/2;

            Log("(%g, %g)  (%g, %g): xscale %g. zscale %g. q(%g, %g")
                    % x1 % y1 % x2 % y2 % c.xscale % c.zscale % q[0] % q[2] ;
            break;
        }
        default:
            break;
        }
    }
    else if (prevPress1 || prevPress2)
    {
        // was rescaling or panning but isn't rescaling nor panning anymore, start over
        press1 = press2 = false;
        mode = Undefined;
    }

    if (q[0] < 0) q[0] = 0;
    if (q[2] < 0) q[2] = 0;
    if (q[2] > 1) q[2] = 1;
    if (r[0] < 0) r[0] = 0;
    if (r[0] > 90) r[0] = 90;
    if (c.zscale < 5) c.zscale = 5; // don't zoom out too much
    if (c.zscale > 100) c.zscale = 100; // don't zoom in too much

    if (c.xscale < 0.01) c.xscale = 0.01; // don't zoom out too much
    if (c.xscale > 5) c.xscale = 5; // don't zoom in too much
    render_model->camera.orthoview = r[0] == 90;
    *render_model->camera.write () = c;

    if (!prevPress1 && press1)
        hstart1 = hpos1, start1 = point1;
    if (!prevPress2 && press2)
        hstart2 = hpos2, start2 = point2;
    prevPress1 = press1;
    prevPress2 = press2;
    prev1 = point1;
    prev2 = point2;

    emit refresh ();
}
