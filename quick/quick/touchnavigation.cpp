#include "touchnavigation.h"
#include "squircle.h"
#include "tools/support/renderviewinfo.h"
#include "log.h"
#include "demangle.h"

//#define LOG_NAVIGATION
#define LOG_NAVIGATION if(0)

using Tools::Support::RenderViewInfo;

TouchNavigation::TouchNavigation(QQuickItem* parent)
    :
      QQuickItem(parent)
{
    connect(&timer,SIGNAL(timeout()),this,SLOT(startSelection()));
    timer.setSingleShot (true);
}


void TouchNavigation::
        setSquircle(Squircle*s)
{
    disconnect (this, SIGNAL(refresh()));

    squircle_=s;
    emit squircleChanged ();

    connect(this, SIGNAL(refresh()), squircle_, SIGNAL(cameraChanged()));
    connect(this, SIGNAL(refresh()), squircle_, SIGNAL(refresh()));
}


void TouchNavigation::
        setSelection(Selection*s)
{
    selection_=s;
    emit selectionChanged ();
}


Tools::RenderModel* TouchNavigation::
        render_model()
{
    return squircle_ ? squircle_->renderModel () : 0;
}


void TouchNavigation::componentComplete()
{
    QQuickItem::componentComplete();

    // qml signals
    connect(this, SIGNAL(mouseMove(qreal,qreal,bool)),
            this, SLOT(onMouseMove(qreal,qreal,bool)));
    connect(this, SIGNAL(touch(qreal,qreal,bool,qreal,qreal,bool,qreal,qreal,bool)),
            this, SLOT(onTouch(qreal,qreal,bool,qreal,qreal,bool,qreal,qreal,bool)));
}


void TouchNavigation::
        onMouseMove(qreal x1, qreal y1, bool p1)
{
    onTouch(x1,y1,p1,0,0,0,0,0,0);
}


void TouchNavigation::
        onTouch(qreal x1, qreal y1, bool press1, qreal x2, qreal y2, bool press2, qreal x3, qreal y3, bool press3)
{
    LOG_NAVIGATION {
        if (!press2 && !press3)
            Log("touchnavigation: %d (%g %g)")
                % press1 % x1 % y1;
        else if (!press3)
            Log("touchnavigation: %d %d (%g %g) (%g %g)")
                % press1 % press2 % x1 % y1 % x2 % y2;
        else
            Log("touchnavigation: %d %d %d (%g %g) (%g %g) (%g %g)")
                % press1 % press2 % press3 % x1 % y1 % x2 % y2 % x3 % y3;
    }

    if (!render_model())
    {
        Log("touchnavigation: doesn't have a render_model");
        return;
    }

    if (press3)
        press1 = press2 = press3 = false;

    QPointF point1(x1,y1), point2(x2,y2);
//    point1 = mapToScene (point1);
//    point2 = mapToScene (point2);
//    x1 = point1.x ();
//    y1 = point1.y ();
//    x2 = point2.x ();
//    y2 = point2.y ();

    Heightmap::Position hpos1, hpos2;

    auto c = *render_model()->camera.read();
    vectord& q = c.q;
    vectord& r = c.r;

    if (press1 && !press2)
        hpos1 = RenderViewInfo(render_model()).getPlanePos (point1);
    else if (press1 && press2)
        hpos1 = RenderViewInfo(render_model()).getPlanePos ((point1+point2)/2);

    if (selection_) {
        if (!prevPress1 && press1 && !press2 && !press3) {
            // this might be a press and hold
            timer.start (600);
        } else if (is_hold && press1) {
            // is hold-selecting
            Heightmap::FreqAxis f = render_model()->tfr_mapping ().read ()->display_scale();
            selection_->setT2 (hpos1.time);
            selection_->setF2 (f.getFrequency(hpos1.scale));
        } else {
            if (timer.isActive ())
                // moved or used more touch points, abort
                timer.stop ();

            if (is_hold)
            {
                is_hold = false;
                emit isHoldChanged();
            }
        }
    }

    if (press1 && !press2 && prevPress1 && !prevPress2)
    {
        if (!is_hold) { // shouldn't navigate while hold-selecting
            // pan is based on the idea that a touched position should still point
            // at the same position when its moving, hence the delta here is not how much the point
            // has moved since the last step but how wrong the current position is compared to the
            // starting position
            double dtime1 = hpos1.time - hstart1.time;
            double dscale1 = hpos1.scale - hstart1.scale;
            q[0] -= dtime1/2;
            q[2] -= dscale1/2;

            LOG_NAVIGATION Log("touchnavigation: (%g, %g)->(%g, %g): q[0] %g. q[2] %g") % x1 % y1 % hpos1.time % hpos1.scale % q[0] % q[2];
        }
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

#ifdef Q_OS_IOS
            r[1] += (dx1 + dx2)/6;
            r[0] += (dy1 + dy2)/6;
#else
            r[1] += (dx1 + dx2)/2;
            r[0] += (dy1 + dy2)/2;
#endif

            LOG_NAVIGATION Log("touchnavigation: (%g, %g)  (%g, %g): r[0] %g. r[1] %g")
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
            //double dtime1 = hpos1.time - hstart1.time;
            //double dscale1 = hpos1.scale - hstart1.scale;
            //q[0] -= dtime1/2;
            //q[2] -= dscale1/2;

            LOG_NAVIGATION Log("touchnavigation: (%g, %g)  (%g, %g): xscale %g. zscale %g. q(%g, %g")
                    % x1 % y1 % x2 % y2 % c.xscale % c.zscale % q[0] % q[2] ;
            break;
        }
        default:
            break;
        }
    }
    else if (prevPress1 || prevPress2)
    {
        LOG_NAVIGATION Log("touchnavigation: start over");
        // was rescaling or panning but isn't rescaling nor panning anymore, start over
        press1 = press2 = false;
        mode = Undefined;
    }

    // match zmin/zmax with OptimalTimeFrequencyResolution
    auto viewport = render_model()->gl_projection.read ()->viewport;
    float aspect = viewport[2]/(float)viewport[3];
    float zmin = std::min(0.5,0.4/(c.zscale/-c.p[2]*aspect));
    float zmax = 1-zmin;

    // limit camera position along scale and limit rotation
    if (q[2] < zmin) q[2] = zmin;
    if (q[2] > zmax) q[2] = zmax;
    if (r[0] < 0) r[0] = 0;
    if (r[0] > 90) r[0] = 90;

    // read info about current view
    auto tm = render_model()->tfr_mapping ().read ();
    double L = tm->length();
    tm.unlock ();

    // limit camera position along time
    if (q[0] < 0) q[0] = 0;
    if (q[0] > L) q[0] = L;

    // limit zoom-in and zoome-out
    float z = -c.p[2];
    c.zscale = std::max(c.zscale, 0.5f*z); // don't zoom out too much
    c.zscale = std::min(c.zscale, z*100); // don't zoom in too much

    c.xscale = std::max(c.xscale, 0.5f*float(z/std::max(1.,L))); // don't zoom out too much
    c.xscale = std::min(c.xscale, z*100); // don't zoom in too much
    c.orthoview = r[0] == 90;

    // update the camera atomically
    *render_model()->camera.write () = c;

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


void TouchNavigation::
        startSelection()
{
    if (selection_) {
        Heightmap::FreqAxis f = render_model()->tfr_mapping ().read ()->display_scale();
        // figure out the resolution where clicked
        vectord p(hstart1.time, 0, hstart1.scale);
        vectord::T timePerPixel;
        vectord::T scalePerPixel;
        glProjection gl_projection = *render_model ()->gl_projection.read ();
        gl_projection.computeUnitsPerPixel( p, timePerPixel, scalePerPixel );

        float dt1 = std::fabs(hstart1.time - selection_->t1 ());
        float dt2 = std::fabs(hstart1.time - selection_->t2 ());
        float ds1 = std::fabs(hstart1.scale - f.getFrequencyScalar (selection_->f1 ()));
        float ds2 = std::fabs(hstart1.scale - f.getFrequencyScalar (selection_->f2 ()));

        int wh = std::min(gl_projection.viewport[2], gl_projection.viewport[3]);
        int threshold = wh/8;

        if (selection_->valid () && std::min(dt1,dt2) < timePerPixel*threshold && std::min(ds1,ds2) < scalePerPixel*threshold)
        {
            // continue on previous selection
            if (dt2>dt1)
                selection_->setT1 (selection_->t2 ());
            if (ds2>ds1)
                selection_->setF1 (selection_->f2 ());
        }
        else
        {
            // start over, new degenerate selection
            selection_->setT1 (hstart1.time);
            selection_->setF1 (f.getFrequency (hstart1.scale));
        }

        selection_->setT2 (hstart1.time);
        selection_->setF2 (f.getFrequency (hstart1.scale));

        is_hold = true;
        emit isHoldChanged();
        emit refresh ();
    }
}
