#include "drawwaveform.h"

// gpumisc
#include "gl.h"
#include "TaskTimer.h"

// Boost
#include <boost/foreach.hpp>

namespace Tools {
    namespace Support {

DrawWaveform::
        DrawWaveform()
            :
            _enqueueGcDisplayList( false )
{
}


void DrawWaveform::
        drawWaveform_chunk_directMode( Signal::pBuffer chunk)
{
    TaskTimer tt(__FUNCTION__);
    DataStorageSize n = chunk->getChannel (0)->waveform_data()->size();
    const float* data = chunk->getChannel (0)->waveform_data()->getCpuMemory();

    n.height = 1;
    float ifs = 1./chunk->sample_rate(); // step per sample
    /*    float max = 1e-6;
     //for (unsigned c=0; c<n.height; c++)
     {
     unsigned c=0;
     for (unsigned t=0; t<n.width; t++)
     if (fabsf(data[t + c*n.width])>max)
     max = fabsf(data[t + c*n.width]);
     }
     float s = 1/max;
     */
    float s = 1;
    //glEnable(GL_BLEND);
    glDepthMask(false);
    //glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    int c=0;
    //    for (unsigned c=0; c<n.height; c++)
    {
        glBegin(GL_TRIANGLE_STRIP);
        //glBegin(GL_POINTS);
        for (int t=0; t<n.width; t+=std::max( 1, (n.width/2000) )) {
            /*float lmin,lmax = (lmin = data[t + c*n.width]);
             for (unsigned j=0; j<std::max((size_t)2, (n.width/1000)) && t<n.width;j++, t++) {
             const float &a = data[t + c*n.width];
             if (a<lmin) lmin=a;
             if (a>lmax) lmax=a;
             }
             glVertex3f( ifs*t, 0, s*lmax);
             glVertex3f( ifs*t, 0, s*lmin);*/
            glVertex3f( ifs*t, 0, s*data[t + c*n.width]);
            float pt = t;
            t+=std::max( 1, n.width/2000 );
            if (t<n.width)
                glVertex3f( ifs*pt, 0, s*data[t + c*n.width]);
        }
        glEnd();
        //        glTranslatef(0, 0, -.5); // different channels along y
    }

    glDepthMask(true);
    //glDisable(GL_BLEND);
}


/**
 draw_glList renders 'chunk' by passing it as argument to 'renderFunction' and caches the results in an OpenGL display list.
 When draw_glList is called again with the same 'chunk' it will not call 'renderFunction' but instead draw the previously cached results.
 If 'force_redraw' is set to true, 'renderFunction' will be called again to replace the old cache.
 */
template<typename RenderData>
void DrawWaveform::
        draw_glList( boost::shared_ptr<RenderData> chunk, void (*renderFunction)( boost::shared_ptr<RenderData> ), bool force_redraw )
{
    // do a cache lookup
    std::map<void*, ListCounter>::iterator itr = _chunkGlList.find(chunk.get());

    if (_chunkGlList.end() == itr && force_redraw) {
        force_redraw = false;
    }

    // cache miss or force_redraw
    if (_chunkGlList.end() == itr || force_redraw) {
        ListCounter cnt;
        if (force_redraw) {
            cnt = itr->second;
            cnt.age = ListCounter::Age_InUse;
        } else {
            cnt.age = ListCounter::Age_JustCreated;
            cnt.displayList = glGenLists(1);
        }

        if (0 != cnt.displayList) {
            glNewList(cnt.displayList, GL_COMPILE_AND_EXECUTE );
            renderFunction( chunk );
            glEndList();
            _chunkGlList[chunk.get()] = cnt;

        } else {
            // render anyway, but not into display list and enqueue gc
            _enqueueGcDisplayList = true;
            renderFunction( chunk );
        }

    } else {
        // render cache
        itr->second.age = ListCounter::Age_InUse; // don't remove

        glCallList( itr->second.displayList );
    }
}


void DrawWaveform::
        gcDisplayList()
{
    /* remove those display lists that haven't been used since last gc
     (used by draw_glList) */
    for (std::map<void*, ListCounter>::iterator itr = _chunkGlList.begin();
         _chunkGlList.end() != itr;
         ++itr)
    {
        if (ListCounter::Age_ProposedForRemoval == itr->second.age) {
            glDeleteLists( itr->second.displayList, 1 );
            _chunkGlList.erase(itr);
            /* restart for-loop as iterators are invalidated by 'erase' */
            itr = _chunkGlList.begin();
        }
    }

    /* at next gc, remove those that haven't been used since this gc */
    typedef std::pair<void* const,ListCounter> lcp;
    BOOST_FOREACH( lcp& cnt, _chunkGlList)
    {
        /*    for (std::map<Spectrogram_chunk*, ListCounter>::iterator itr = _chunkGlList.begin();
         _chunkGlList.end() != itr;
         ++itr)
         {*/
        cnt.second.age = ListCounter::Age_ProposedForRemoval;
    }

    _enqueueGcDisplayList = false;
}

} // namespace Support
} // namespace Tools
