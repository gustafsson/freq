varying highp float vertex_height;
varying mediump float shadow;
varying highp vec2 texCoord;

uniform highp sampler2D tex;
uniform mediump sampler2D tex_color;
uniform mediump float colorTextureFactor;
uniform lowp float contourPlot;
uniform mediump float yScale;
uniform mediump float yOffset;
uniform lowp float yNormalize;
uniform mediump vec3 logScale;
uniform lowp vec4 fixedColor;
uniform lowp vec4 clearColor;
uniform mediump vec2 texSize;

#ifdef USE_MIPMAP
varying mediump float vbias;
uniform highp sampler2D tex_ota;
#endif

mediump float heightValue(mediump float v) {
//    v *= 0.01; // map from small range of float16 to float32 (TfrBlockUpdater)
    mediump float h = mix(v*yScale + yOffset,       // linear
                  log(v) * logScale.y + logScale.z, // log
                  logScale.x);                      // choose

    return v == 0.0 ? 0.0 : max(0.0, h);
}


#ifdef USE_MIPMAP
// https://www.opengl.org/discussion_boards/showthread.php/177520-Mipmap-level-calculation-using-dFdx-dFdy
// returns the mipmap level to use
// returns<0 when magnifying
// returns>0 when minifying
mediump float mipmap_level(mediump vec2 texture_coordinate)
{
    // The OpenGL Graphics System: A Specification 4.2
    //  - chapter 3.9.11, equation 3.21

    mediump vec2  dx_vtc        = dFdx(texture_coordinate);
    mediump vec2  dy_vtc        = dFdy(texture_coordinate);
    mediump float delta_max_sqr = max(dot(dx_vtc, dx_vtc), dot(dy_vtc, dy_vtc));


    //return max(0.0, 0.5 * log2(delta_max_sqr) - 1.0); // == log2(sqrt(delta_max_sqr));
    return 0.5 * log2(delta_max_sqr); // == log2(sqrt(delta_max_sqr));
}
#endif


void main()
{
    // anti-aliasing, multisample the texture four times per pixel, should use 4 different varyings
    //vec2 dt = 0.25*fwidth(texCoord);
    //vec4 v4 = vec4(texture2D(tex, texCoord + dt).x,
    //        texture2D(tex, texCoord + vec2(dt.x, -dt.y)).x,
    //        texture2D(tex, texCoord - dt).x,
    //        texture2D(tex, texCoord - vec2(dt.x, -dt.y)).x);
    //float v = max(max(v4.x, v4.y), max(v4.z, v4.w));
    //float v = (v4.x + v4.y + v4.z + v4.w) / 4.0;

    mediump float v = texture2D(tex, texCoord).x;
#ifdef USE_MIPMAP
    if (yNormalize>0.0)
    {
        // 1. normalize to the background level.
        // 2. let background level be defined by a rolling median
        // 2. use approximate OTA as an approximation of median
        // 3. given that sharp peaks are way more common than sharp valleys the OTA
        //    is high near a peak. Thus a small value in a lower mipmap level is more
        //    likely to represent the background level.
        mediump float median = texture2DLod(tex_ota, texCoord, 4.0+vbias).x;
//                min(texture2DLod(tex_ota, texCoord, 4.0+vbias).x,
//                    texture2DLod(tex_ota, texCoord, 2.0+vbias).x);

#define scalefactor 0.2
        v = mix(v, scalefactor*(1.0 + (v-median)/median), yNormalize);
    }
#endif

    // up until here, 'v' doesn't depend on any render settings (uniforms) and could be
    // precomputed instead (apart from varying projections causing different mipmap
    // resolutions, but let's disregard that)

//    v = mix(clamp(v*0.005,0.0,1.0),heightValue(v),colorTextureFactor);
    v = heightValue(v);
    mediump float height_value = v;

    // rainbow, colorscale or grayscale
    mediump vec4 curveColor = mix(fixedColor, texture2D(tex_color, vec2(v,0)), colorTextureFactor);
    mediump float g = max(0.0, 1.0-v);
    v = mix(v, 1.0 - g*g*g, colorTextureFactor);

    //float fresnel   = pow(1.0 - facing, 5.0); // Fresnel approximation
#ifdef DRAW3D
#ifndef NOSHADOW
    curveColor *= shadow; // curveColor*shadow + vec4(fresnel);
#endif
#endif
    curveColor = mix(clearColor, curveColor, v);

#ifdef DRAWISARITHM
#ifdef DRAW3D
    mediump float isarithm1 = fract( vertex_height * 25.0) < 0.93 ? 1.0 : 0.8;
    mediump float isarithm2 = fract( vertex_height * 5.0) < 0.93 ? 1.0 : 0.8;
    curveColor = mix( curveColor, curveColor* isarithm1 * isarithm2*isarithm2, contourPlot);
#else
    mediump float isarithm1 = fract( height_value * 25.0) < 0.93 ? 1.0 : 0.8;
    mediump float isarithm2 = fract( height_value * 5.0) < 0.93 ? 1.0 : 0.8;
    curveColor = mix( curveColor, curveColor* isarithm1 * isarithm2*isarithm2, contourPlot);
#endif
#endif

//    curveColor.w = 1.0; //-saturate(fresnel);
    gl_FragColor = max(min(curveColor, 1.0), 0.0);
}
