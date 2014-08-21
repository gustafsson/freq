varying highp float vertex_height;
varying mediump float shadow;
varying highp vec2 texCoord;

uniform highp sampler2D tex;
uniform mediump sampler2D tex_color;
uniform mediump float colorTextureFactor;
uniform lowp float contourPlot;
uniform mediump float yScale;
uniform mediump float yOffset;
uniform mediump vec3 logScale;
uniform lowp vec4 fixedColor;
uniform lowp vec4 clearColor;

mediump float heightValue(mediump float v) {
    // the linear case is straightforward
    mediump float h = mix(v*yScale + yOffset,
                  log(v) * logScale.y + logScale.z,
                  logScale.x);

    return v == 0.0 ? 0.0 : max(0.01, h);
}

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
    v = heightValue(v);

    mediump vec4 curveColor = fixedColor; // colorscale or grayscale

    mediump float f = v;

    // rainbow
    curveColor = mix(curveColor, texture2D(tex_color, vec2(f,0)), colorTextureFactor);
    mediump float g = max(0.0, 1.0-f);
    f = mix(f, 1.0 - g*g*g, colorTextureFactor);

    //float fresnel   = pow(1.0 - facing, 5.0); // Fresnel approximation
    curveColor *= shadow; // curveColor*shadow + vec4(fresnel);
    curveColor = mix(clearColor, curveColor, f);

    mediump float isarithm1 = fract( vertex_height * 25.0) < 0.93 ? 1.0 : 0.8;
    mediump float isarithm2 = fract( vertex_height * 5.0) < 0.93 ? 1.0 : 0.8;
    curveColor = mix( curveColor, curveColor* isarithm1 * isarithm2*isarithm2, contourPlot);

    curveColor.w = 1.0; //-saturate(fresnel);
    gl_FragColor = curveColor;
}
