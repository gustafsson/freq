#include "renderaxes.h"
#include "frustumclip.h"

// gpumisc
#include "tasktimer.h"
#include "glPushContext.h"
#include "gl.h"

// glut
#ifndef __APPLE__
#   include <GL/glut.h>
#else
# ifndef GL_ES_VERSION_2_0
#   include <GLUT/glut.h>
# endif
#endif

//#define TIME_RENDERER
#define TIME_RENDERER if(0)

namespace Heightmap {
namespace Render {

RenderAxes::
        RenderAxes(
                const RenderSettings& render_settings,
                const glProjection* gl_projection,
                FreqAxis display_scale)
    :
      render_settings(render_settings),
      gl_projection(gl_projection),
      display_scale(display_scale)
{
    // Using glut for drawing fonts, so glutInit must be called.
    static int c=0;
    if (0==c)
    {
        // run glutinit once per process
#ifdef _WIN32
        c = 1;
        char* dummy="dummy\0";
        glutInit(&c,&dummy);
#elif !defined(__APPLE__)
        glutInit(&c,0);
        c = 1;
#endif
    }
}


void RenderAxes::
        drawAxes( float T )
{
#ifndef GL_ES_VERSION_2_0
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf (GLmatrixf(gl_projection->projection).v ());
    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf (GLmatrixf(gl_projection->modelview).v ());

    TIME_RENDERER TaskTimer tt("drawAxes(length = %g)", T);
    // Draw overlay borders, on top, below, to the right or to the left
    // default left bottom

    // 1 gray draw overlay
    // 2 clip entire sound to frustum
    // 3 decide upon scale
    // 4 draw axis
    const glProjection* g = gl_projection;
    unsigned screen_width = g->viewport[2];
    unsigned screen_height = g->viewport[3];

    float borderw = 12.5*1.1;
    float borderh = 12.5*1.1;

    float scale = render_settings.dpifactor;
    borderw *= scale;
    borderh *= scale;

    float w = borderw/screen_width, h=borderh/screen_height;
    Render::FrustumClip frustum_clip(*g, w, h);

    if (render_settings.axes_border) { // 1 gray draw overlay
        glPushMatrixContext push_model(GL_MODELVIEW);
        glPushMatrixContext push_proj(GL_PROJECTION);

        glLoadIdentity();
        gluOrtho2D( 0, 1, 0, 1 );

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glDisable(GL_DEPTH_TEST);


        glColor4f( 1.0f, 1.0f, 1.0f, .4f );
        glBegin( GL_QUADS );
            glVertex2f( 0, 0 );
            glVertex2f( w, 0 );
            glVertex2f( w, 1 );
            glVertex2f( 0, 1 );

            glVertex2f( w, 0 );
            glVertex2f( 1-w, 0 );
            glVertex2f( 1-w, h );
            glVertex2f( w, h );

            glVertex2f( 1, 0 );
            glVertex2f( 1, 1 );
            glVertex2f( 1-w, 1 );
            glVertex2f( 1-w, 0 );

            glVertex2f( w, 1 );
            glVertex2f( w, 1-h );
            glVertex2f( 1-w, 1-h );
            glVertex2f( 1-w, 1 );
        glEnd();

        glEnable(GL_DEPTH_TEST);

        //glDisable(GL_BLEND);
    }

    // 2 clip entire sound to frustum
    std::vector<vectord> clippedFrustum;

    vectord closest_i;
    {   //float T = collection->worker->source()->length();
        vectord corner[4]=
        {
            vectord( 0, 0, 0),
            vectord( 0, 0, 1),
            vectord( T, 0, 1),
            vectord( T, 0, 0),
        };

        clippedFrustum = frustum_clip.clipFrustum (corner, &closest_i);
    }


    // 3 find inside
    vectord inside;
    {
        for (unsigned i=0; i<clippedFrustum.size(); i++)
            inside = inside + clippedFrustum[i];

        // as clippedFrustum is a convex polygon, the mean position of its vertices will be inside
        inside = inside * (1.f/clippedFrustum.size());
    }


    // 4 render and decide upon scale
    vectord x(1,0,0), z(0,0,1);

    glDepthMask(false);
    glDisable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    FreqAxis fa = display_scale;
    // loop along all sides
    typedef tvector<4,GLfloat> GLvectorF;
    typedef tvector<2,GLfloat> GLvector2F;
    std::vector<GLvectorF> ticks;
    std::vector<GLvectorF> phatTicks;
    std::vector<GLvector2F> quad(4);

    for (unsigned i=0; i<clippedFrustum.size(); i++)
    {
        glColor4f(0,0,0,0.8);
        unsigned j=(i+1)%clippedFrustum.size();
        vectord p1 = clippedFrustum[i]; // starting point of side
        vectord p2 = clippedFrustum[j]; // end point of side
        vectord v0 = p2-p1;

        // decide if this side is a t or f axis
        vectord::T timePerPixel, scalePerPixel;
        g->computeUnitsPerPixel( inside, timePerPixel, scalePerPixel );
        timePerPixel *= scale; scalePerPixel *= scale;

        bool taxis = fabsf(v0[0]*scalePerPixel) > fabsf(v0[2]*timePerPixel);


        // decide in which direction to traverse this edge
        vectord::T timePerPixel1, scalePerPixel1, timePerPixel2, scalePerPixel2;
        g->computeUnitsPerPixel( p1, timePerPixel1, scalePerPixel1 );
        g->computeUnitsPerPixel( p2, timePerPixel2, scalePerPixel2 );
        timePerPixel1 *= scale; scalePerPixel1 *= scale;
        timePerPixel2 *= scale; scalePerPixel2 *= scale;

        double dscale = 0.001;
        double hzDelta1= fabs(fa.getFrequencyT( p1[2] + v0[2]*dscale ) - fa.getFrequencyT( p1[2] ));
        double hzDelta2 = fabs(fa.getFrequencyT( p2[2] - v0[2]*dscale ) - fa.getFrequencyT( p2[2] ));

        if ((taxis && timePerPixel1 > timePerPixel2) || (!taxis && hzDelta1 > hzDelta2))
        {
            vectord flip = p1;
            p1 = p2;
            p2 = flip;
        }

        vectord p = p1; // starting point
        vectord v = p2-p1;

        if (!v[0] && !v[2]) // skip if |v| = 0
            continue;


        vectord::T timePerPixel_closest, scalePerPixel_closest;
        g->computeUnitsPerPixel( closest_i, timePerPixel_closest, scalePerPixel_closest );
        timePerPixel_closest *= scale; scalePerPixel_closest *= scale;

        if (render_settings.draw_axis_at0==-1)
        {
            (taxis?p[2]:p[0]) = ((taxis?p[2]:p[0])==0) ? 1 : 0;
            (taxis?p1[2]:p1[0]) = ((taxis?p1[2]:p1[0])==0) ? 1 : 0;
        }

        // need initial f value
        vectord pp = p;
        double f = fa.getFrequencyT( p[2] );

        if (((taxis && render_settings.draw_t) || (!taxis && render_settings.draw_hz)) &&
            (render_settings.draw_axis_at0!=0?(taxis?p[2]==0:p[0]==0):true))
        for (double u=-1; true; )
        {
            vectord::T timePerPixel, scalePerPixel;
            g->computeUnitsPerPixel( p, timePerPixel, scalePerPixel );
            timePerPixel *= scale; scalePerPixel *= scale;

            double ppp=0.4;
            timePerPixel = timePerPixel * ppp + timePerPixel_closest * (1.0-ppp);
            scalePerPixel = scalePerPixel * ppp + scalePerPixel_closest * (1.0-ppp);

            vectord::T ST = timePerPixel * 750; // ST = time units per 750 pixels, 750 pixels is a fairly common window size
            vectord::T SF = scalePerPixel * 750;
            double drawScaleT = std::min(ST, 50*timePerPixel_closest*750);
            double drawScaleF = std::min(SF, 50*scalePerPixel_closest*750);

            double time_axis_density = 18;
            if (20.+2.*log10(timePerPixel) < 18.)
                time_axis_density = std::max(1., 20.+2.*log10(timePerPixel));

            double scale_axis_density = std::max(10., 22. - ceil(fabs(log10(f))));
            if (f == 0)
                scale_axis_density = 21;

            int st = floor(log10( ST / time_axis_density ));
            //int sf = floor(log10( SF / scale_axis_density ));

            double DT = pow(10.0, st);
            //double DF = pow(10, sf);
            double DF = std::min(0.2, SF / scale_axis_density);

            // compute index of next marker along t and f
            int tmultiple = 10, tsubmultiple = 5;

            if (st>0)
            {
                st = 0;
                if( 60 < ST ) DT = 10, st++, tmultiple = 6, tsubmultiple = 3;
                if( 60*10 < ST ) DT *= 6, st++, tmultiple = 10, tsubmultiple = 5;
                if( 60*10*6 < ST ) DT *= 10, st++, tmultiple = 6, tsubmultiple = 3;
                if( 60*10*6*24 < ST ) DT *= 6, st++, tmultiple = 24, tsubmultiple = 6;
                if( 60*10*6*24*5 < ST ) DT *= 24, st++, tmultiple = 5, tsubmultiple = 5;
            }

            int tupdatedetail = 1;
            DT /= tupdatedetail;
            int t = floor(p[0]/DT + .5); // t marker index along t
            double t2 = t*DT;
            if (t2 < p[0] && v[0] > 0)
                t++;
            if (t2 > p[0] && v[0] < 0)
                t--;
            p[0] = t*DT;

            //int tmarkanyways = (bool)(fabsf(5*DT*tupdatedetail) > (ST / time_axis_density) && ((unsigned)(p[0]/DT + 0.5)%(tsubmultiple*tupdatedetail)==0) && ((unsigned)(p[0]/DT +.5)%(tmultiple*tupdatedetail)!=0));
            int tmarkanyways = (bool)(fabsf(5*DT*tupdatedetail) > (ST / time_axis_density) && ((unsigned)(p[0]/DT + 0.5)%(tsubmultiple*tupdatedetail)==0));
            if (tmarkanyways)
                st--;

            // compute index of next marker along t and f
            double epsilon = 1.f/10;
            double hz1 = fa.getFrequencyT( p[2] - DF * epsilon );
            double hz2 = fa.getFrequencyT( p[2] + DF * epsilon );
            if (hz2-f < f-hz1)  hz1 = f;
            else                hz2 = f;
            double fc0 = (hz2 - hz1)/epsilon;
            int sf = floor(log10( fc0 ));
            double fc = powf(10, sf);
            int fmultiple = 10;

            int fupdatedetail = 1;
            fc /= fupdatedetail;
            int mif = floor(f / fc + .5); // f marker index along f
            double nf = mif * fc;
            if (!(f - fc*0.05 < nf && nf < f + fc*0.05))
                nf += v[2] > 0 ? fc : -fc;
            f = nf;
            p[2] = fa.getFrequencyScalarNotClampedT(f);

            double np1 = fa.getFrequencyScalarNotClampedT( f + fc);
            double np2 = fa.getFrequencyScalarNotClampedT( f - fc);
            int fmarkanyways = false;
            fmarkanyways |= 0.9*fabsf(np1 - p[2]) > DF && 0.9*fabsf(np2 - p[2]) > DF && ((unsigned)(f / fc + .5)%1==0);
            fmarkanyways |= 4.5*fabsf(np1 - p[2]) > DF && 4.5*fabsf(np2 - p[2]) > DF && ((unsigned)(f / fc + .5)%5==0);
            if (fmarkanyways)
                sf--;


            if (taxis && render_settings.draw_cursor_marker)
            {
                float w = (render_settings.cursor[0] - pp[0])/(p[0] - pp[0]);

                if (0 < w && w <= 1)
                    if (!tmarkanyways)
                        st--;

                if (fabsf(w) < tupdatedetail*tmultiple/2)
                    tmarkanyways = -1;

                if (0 < w && w <= 1)
                {
                    DT /= 10;
                    t = floor(render_settings.cursor[0]/DT + 0.5); // t marker index along t

                    p = p1 + v*((render_settings.cursor[0] - p1[0])/v[0]);
                    p[0] = render_settings.cursor[0]; // exact float value so that "cursor[0] - pp[0]" == 0

                    if (!tmarkanyways)
                        st--;

                    tmarkanyways = 2;
                }
            }
            else if(render_settings.draw_cursor_marker && fa.axis_scale != AxisScale_Unknown)
            {
                float w = (render_settings.cursor[2] - pp[2])/(p[2] - pp[2]);

                if (0 < w && w <= 1)
                    if (!fmarkanyways)
                        sf--;

                if (fabsf(w) < fupdatedetail*fmultiple/2)
                    fmarkanyways = -1;

                if (0 < w && w <= 1)
                {
                    f = fa.getFrequencyT( render_settings.cursor[2] );
                    fc /= 10;
                    mif = floor(f / fc + .5); // f marker index along f
                    f = mif * fc;

                    p = p1 + v*((render_settings.cursor[2] - p1[2])/v[2]);
                    p[2] = render_settings.cursor[2]; // exact float value so that "cursor[2] - pp[2]" == 0

                    fmarkanyways = 2;
                }
            }


            // find next intersection along v
            double nu;
            int c1 = taxis ? 0 : 2;
            int c2 = !taxis ? 0 : 2;
            nu = (p[c1] - p1[c1])/v[c1];

            // if valid intersection
            if ( nu > u && nu<=1 ) { u = nu; }
            else break;

            // compute intersection
            p[c2] = p1[c2] + v[c2]*u;


            vectord np = p;
            nf = f;
            int nt = t;

            if (taxis)
            {
                if (v[0] > 0) nt++;
                if (v[0] < 0) nt--;
                np[0] = nt*DT;
            }
            else
            {
                if (v[2] > 0) nf+=fc;
                if (v[2] < 0) nf-=fc;
                nf = floor(nf/fc + .5)*fc;
                np[2] = fa.getFrequencyScalarNotClampedT(nf);
            }

            // draw marker
            if (taxis) {
                if (0 == t%tupdatedetail || tmarkanyways==2)
                {
                    float size = 1+ (0 == (t%(tupdatedetail*tmultiple)));
                    if (tmarkanyways)
                        size = 2;
                    if (-1 == tmarkanyways)
                        size = 1;

                    float sign = (v0^z)%(v0^( p - inside))>0 ? 1.f : -1.f;
                    float o = size*SF*.003f*sign;

                    (size==1?ticks:phatTicks).push_back(GLvectorF(p[0], 0.f, p[2], 1.f));
                    (size==1?ticks:phatTicks).push_back(GLvectorF(p[0], 0.f, p[2]-o, 1.f));

                    if (size>1) {
                        glPushMatrixContext push_model( GL_MODELVIEW );

                        glTranslatef(p[0], 0, p[2]);
                        glRotatef(90,1,0,0);
                        glScalef(0.00013f*drawScaleT,0.00013f*drawScaleF,1.f);
                        float angle = atan2(v0[2]/SF, v0[0]/ST) * (180*M_1_PI);
                        glRotatef(angle,0,0,1);
                        char a[100];
                        char b[100];
                        sprintf(b,"%%d:%%0%d.%df", 2+(st<-1?-st:0), st<0?-1-st:0);
                        int minutes = (int)(t*DT/60);
                        sprintf(a, b, minutes,t*DT-60*minutes);
                        float w=0;
                        float letter_spacing=15;

                        for (char*c=a;*c!=0; c++) {
                            if (c!=a)
                                w+=letter_spacing;
                            w+=glutStrokeWidth( GLUT_STROKE_ROMAN, *c );
                        }

                        if (!render_settings.left_handed_axes)
                            glScalef(-1,1,1);
                        glTranslatef(0,70.f,0);
                        if (sign<0)
                            glRotatef(180,0,0,1);
                        glTranslatef(-.5f*w,-50.f,0);
                        glColor4f(1,1,1,0.5);
                        float z = 10;
                        float q = 20;
                        glEnableClientState(GL_VERTEX_ARRAY);
                        quad[0] = GLvector2F(0 - z, 0 - q);
                        quad[1] = GLvector2F(w + z, 0 - q);
                        quad[2] = GLvector2F(0 - z, 100 + q);
                        quad[3] = GLvector2F(w + z, 100 + q);
                        glVertexPointer(2, GL_FLOAT, 0, &quad[0]);
                        glDrawArrays(GL_TRIANGLE_STRIP, 0, quad.size());
                        glDisableClientState(GL_VERTEX_ARRAY);
                        glColor4f(0,0,0,0.8);
                        for (char*c=a;*c!=0; c++) {
                            glutStrokeCharacter(GLUT_STROKE_ROMAN, *c);
                            glTranslatef(letter_spacing,0,0);
                        }
                    }
                }
            } else if (fa.axis_scale != AxisScale_Unknown) {
                if (0 == ((unsigned)floor(f/fc + .5))%fupdatedetail || fmarkanyways==2)
                {
                    int size = 1;
                    if (0 == ((unsigned)floor(f/fc + .5))%(fupdatedetail*fmultiple))
                        size = 2;
                    if (fmarkanyways)
                        size = 2;
                    if (-1 == fmarkanyways)
                        size = 1;


                    float sign = (v0^x)%(v0^( p - inside))>0 ? 1.f : -1.f;
                    float o = size*ST*.003f*sign;

                    (size==1?ticks:phatTicks).push_back(GLvectorF(p[0], 0.f, p[2], 1.f));
                    (size==1?ticks:phatTicks).push_back(GLvectorF(p[0]-o, 0.f, p[2], 1.f));


                    if (size>1)
                    {
                        glPushMatrixContext push_model( GL_MODELVIEW );

                        glTranslatef(p[0],0,p[2]);
                        glRotatef(90,1,0,0);
                        glScalef(0.00013f*drawScaleT,0.00013f*drawScaleF,1.f);
                        float angle = atan2(v0[2]/SF, v0[0]/ST) * (180*M_1_PI);
                        glRotatef(angle,0,0,1);
                        char a[100];
                        char b[100];
                        sprintf(b,"%%.%df", sf<0?-1-sf:0);
                        sprintf(a, b, f);
                        //sprintf(a,"%g", f);
                        unsigned w=0;
                        float letter_spacing=5;

                        for (char*c=a;*c!=0; c++)
                        {
                            if (c!=a)
                                w+=letter_spacing;
                            w+=glutStrokeWidth( GLUT_STROKE_ROMAN, *c );
                        }
                        if (!render_settings.left_handed_axes)
                            glScalef(-1,1,1);
                        glTranslatef(0,70.f,0);
                        if (sign<0)
                            glRotatef(180,0,0,1);
                        glTranslatef(-.5f*w,-50.f,0);
                        glColor4f(1,1,1,0.5);
                        float z = 10;
                        float q = 20;
                        glEnableClientState(GL_VERTEX_ARRAY);
                        quad[0] = GLvector2F(0 - z, 0 - q);
                        quad[1] = GLvector2F(w + z, 0 - q);
                        quad[2] = GLvector2F(0 - z, 100 + q);
                        quad[3] = GLvector2F(w + z, 100 + q);
                        glVertexPointer(2, GL_FLOAT, 0, &quad[0]);
                        glDrawArrays(GL_TRIANGLE_STRIP, 0, quad.size());
                        glDisableClientState(GL_VERTEX_ARRAY);
                        glColor4f(0,0,0,0.8);
                        for (char*c=a;*c!=0; c++)
                            glutStrokeCharacter(GLUT_STROKE_ROMAN, *c);
                    }
                }
            }

            pp = p;
            p = np; // q.e.d.
            f = nf;
            t = nt;
        }

        if (!taxis && render_settings.draw_piano && (render_settings.draw_axis_at0?p[0]==0:true))
        {
            vectord::T timePerPixel, scalePerPixel;
            g->computeUnitsPerPixel( p + v*0.5, timePerPixel, scalePerPixel );
            timePerPixel *= scale; scalePerPixel *= scale;

            double ST = timePerPixel * 750;
            // double SF = scalePerPixel * 750;

            // from http://en.wikipedia.org/wiki/Piano_key_frequencies
            // F(n) = 440 * pow(pow(2, 1/12),n-49)
            // log(F(n)/440) = log(pow(2, 1/12),n-49)
            // log(F(n)/440) = log(pow(2, 1/12))*log(n-49)
            // log(F(n)/440)/log(pow(2, 1/12)) = log(n-49)
            // n = exp(log(F(n)/440)/log(pow(2, 1/12))) + 49

            unsigned F1 = fa.getFrequency( (float)p1[2] );
            unsigned F2 = fa.getFrequency( (float)(p1+v)[2] );
            if (F2<F1) { unsigned swap = F2; F2=F1; F1=swap; }
            if (!(F1>fa.min_hz)) F1=fa.min_hz;
            if (!(F2<fa.max_hz())) F2=fa.max_hz();
            float tva12 = powf(2.f, 1.f/12);


            if (0 == F1)
                F1 = 1;
            int startTone = log(F1/440.f)/log(tva12) + 45;
            int endTone = ceil(log(F2/440.f)/log(tva12)) + 45;
            float sign = (v^x)%(v^( clippedFrustum[i] - inside))>0 ? 1.f : -1.f;
            if (!render_settings.left_handed_axes)
                sign *= -1;

            for( int tone = startTone; tone<=endTone; tone++)
            {
                float ff = fa.getFrequencyScalar(440 * pow(tva12,tone-45));
                float ffN = fa.getFrequencyScalar(440 * pow(tva12,tone-44));
                float ffP = fa.getFrequencyScalar(440 * pow(tva12,tone-46));

                int toneTest = tone;
                while(toneTest<0) toneTest+=12;

                bool blackKey = false;
                switch(toneTest%12) { case 1: case 3: case 6: case 8: case 10: blackKey = true; }
                bool blackKeyP = false;
                switch((toneTest+11)%12) { case 1: case 3: case 6: case 8: case 10: blackKeyP = true; }
                bool blackKeyN = false;
                switch((toneTest+1)%12) { case 1: case 3: case 6: case 8: case 10: blackKeyN = true; }
                glLineWidth(1);
                float wN = ffN-ff, wP = ff-ffP;
                if (blackKey)
                    wN *= .5, wP *= .5;
                else {
                    if (!blackKeyN)
                        wN *= .5;
                    if (!blackKeyP)
                        wP *= .5;
                }

                float u = (ff - p1[2])/v[2];
                float un = (ff+wN - p1[2])/v[2];
                float up = (ff-wP - p1[2])/v[2];
                vectord pt = p1+v*u;
                vectord pn = p1+v*un;
                vectord pp = p1+v*up;

                glPushMatrixContext push_model( GL_MODELVIEW );

                float xscale = 0.016000f;
                float blackw = 0.4f;

                if (sign>0)
                    glTranslatef( xscale*ST, 0.f, 0.f );

                tvector<4,GLfloat> keyColor(0,0,0, 0.7f * blackKey);
                if (render_settings.draw_cursor_marker)
                {
                    float w = (render_settings.cursor[2] - ff)/(ffN - ff);
                    w = fabsf(w/1.6f);
                    if (w < 1)
                    {
                        keyColor[1] = (1-w)*(1-w);
                        if (blackKey)
                            keyColor[3] = keyColor[3]*w + .9f*(1-w);
                        else
                            keyColor[3] = keyColor[1] * .7f;
                    }
                }

                if (keyColor[3] != 0)
                {
                    glColor4fv(keyColor.v);
                    if (blackKey)
                    {
                        glBegin(GL_TRIANGLE_STRIP);
                            glVertex3d(pp[0] - xscale*ST*(1.f), 0, pp[2]);
                            glVertex3d(pp[0] - xscale*ST*(blackw), 0, pp[2]);
                            glVertex3d(pn[0] - xscale*ST*(1.f), 0, pn[2]);
                            glVertex3d(pn[0] - xscale*ST*(blackw), 0, pn[2]);
                        glEnd();
                    }
                    else
                    {
                        glBegin(GL_TRIANGLE_FAN);
                            if (blackKeyP)
                            {
                                glVertex3dv((pp*0.5 + pt*0.5 - vectord(xscale*ST*(blackw), 0, 0)).v);
                                glVertex3d(pp[0] - xscale*ST*(blackKeyP ? blackw : 1.f), 0, pp[2]);
                            }
                            glVertex3d(pp[0] - xscale*ST*(0.f), 0, pp[2]);
                            glVertex3d(pn[0] - xscale*ST*(0.f), 0, pn[2]);
                            glVertex3d(pn[0] - xscale*ST*(blackKeyN ? blackw : 1.f), 0, pn[2]);
                            if (blackKeyN)
                            {
                                glVertex3dv((pn*0.5 + pt*0.5 - vectord(xscale*ST*(blackw), 0, 0)).v);
                                glVertex3dv((pn*0.5 + pt*0.5 - vectord(xscale*ST*(1.f), 0, 0)).v);
                            }
                            if (blackKeyP)
                                glVertex3dv((pp*0.5 + pt*0.5 - vectord(xscale*ST*(1.f), 0, 0)).v);
                            else
                                glVertex3d(pp[0] - xscale*ST*(blackKeyP ? blackw : 1.f), 0, pp[2]);
                        glEnd();
                    }
                }

                // outline
                glColor4f(0,0,0,0.8);
                    glBegin(GL_LINES );
                        glVertex3d(pn[0] - xscale*ST, 0, pn[2]);
                        glVertex3d(pp[0] - xscale*ST, 0, pp[2]);
                    glEnd();
                    glBegin(GL_LINE_STRIP);
                        glVertex3d(pp[0] - xscale*ST*(blackKeyP ? blackw : 1.f), 0, pp[2]);
                        glVertex3d(pp[0] - xscale*ST*(blackKey ? blackw : 0.f), 0, pp[2]);
                        glVertex3d(pn[0] - xscale*ST*(blackKey ? blackw : 0.f), 0, pn[2]);
                        glVertex3d(pn[0] - xscale*ST*(blackKeyN ? blackw : 1.f), 0, pn[2]);
                    glEnd();

                glColor4f(0,0,0,0.8);

                if (tone%12 == 0)
                {
                    glLineWidth(1);

                    glPushMatrixContext push_model( GL_MODELVIEW );
                    glTranslatef(pp[0], 0, pp[2]);
                    glRotatef(90,1,0,0);

                    //glScalef(0.00014f*ST,0.00014f*SF,1.f);
                    glScalef(0.005 * xscale*ST,0.35 * xscale*(pn[2]-pp[2]),1.f);

                    char a[100];
                    sprintf(a,"C%d", tone/12+1);
                    float w=10;
                    for (char*c=a;*c!=0; c++)
                        w+=glutStrokeWidth( GLUT_STROKE_ROMAN, *c );
                    if (!render_settings.left_handed_axes)
                        glScalef(-1,1,1);
                    glTranslatef(-w,0,0);
                    glColor4f(1,1,1,0.5);
                    float z = 10;
                    float q = 20;
                    glBegin(GL_TRIANGLE_STRIP);
                        glVertex2f(0 - z, 0 - q);
                        glVertex2f(w + z, 0 - q);
                        glVertex2f(0 - z, 100 + q);
                        glVertex2f(w + z, 100 + q);
                    glEnd();
                    glColor4f(0,0,0,0.9);
                    for (char*c=a;*c!=0; c++)
                        glutStrokeCharacter(GLUT_STROKE_ROMAN, *c);
                }
            }
        }
    }

    glEnableClientState(GL_VERTEX_ARRAY);
    if (!phatTicks.empty())
    {
        glLineWidth(2);
        glVertexPointer(4, GL_FLOAT, 0, &phatTicks[0]);
        glDrawArrays(GL_LINES, 0, phatTicks.size());
        glLineWidth(1);
    }
    if (!ticks.empty())
    {
        glVertexPointer(4, GL_FLOAT, 0, &ticks[0]);
        glDrawArrays(GL_LINES, 0, ticks.size());
    }
    glDisableClientState(GL_VERTEX_ARRAY);

    glEnable(GL_DEPTH_TEST);
    glDepthMask(true);
#endif // GL_ES_VERSION_2_0
}


} // namespace Render
} // namespace Heightmap
