// GLSL fragment shader
//varying vec3 eyeSpacePos;
//varying vec3 worldSpaceNormal;
//varying vec3 eyeSpaceNormal;
varying float vertex_height;
varying float shadow;
varying vec2 texCoord;

uniform sampler2D tex;
uniform sampler2D tex_color;
uniform float colorTextureFactor;
uniform float contourPlot;
uniform float yScale;
uniform float yOffset;
uniform vec3 logScale;
uniform vec4 fixedColor;
uniform vec4 clearColor;

float heightValue(float v) {
    // the linear case is straightforward
    float h = mix(v*yScale + yOffset,
                  log(v) * logScale.y + logScale.z,
                  logScale.x);

    return v == 0.0 ? 0.0 : max(0.01, h);
}

void main()
{
    // anti-aliasing, multiplesample the texture four times per pixel
    //vec2 dt = 0.25*fwidth(texCoord);
    //vec4 v4 = vec4(texture2D(tex, texCoord + dt).x,
    //        texture2D(tex, texCoord + vec2(dt.x, -dt.y)).x,
    //        texture2D(tex, texCoord - dt).x,
    //        texture2D(tex, texCoord - vec2(dt.x, -dt.y)).x);
    //float v = max(max(v4.x, v4.y), max(v4.z, v4.w));
    //float v = (v4.x + v4.y + v4.z + v4.w) / 4.0;

    float v = texture2D(tex, texCoord).x;
    v = heightValue(v);

    vec4 curveColor = fixedColor; // colorscale or grayscale

    float f = v;

    // rainbow
    curveColor = mix(curveColor, texture2D(tex_color, vec2(f,0)), colorTextureFactor);
    float g = max(0.0, 1.0-f);
    f = mix(f, 1.0 - g*g*g, colorTextureFactor);

    //float fresnel   = pow(1.0 - facing, 5.0); // Fresnel approximation
    curveColor *= shadow; // curveColor*shadow + vec4(fresnel);
    curveColor = mix(clearColor, curveColor, f);

    v = 0.0==vertex_height ? v : vertex_height;
    float isarithm1 = fract( v * 25.0) < 0.93 ? 1.0 : 0.8;
    float isarithm2 = fract( v * 5.0) < 0.93 ? 1.0 : 0.8;
    curveColor = mix( curveColor, curveColor* isarithm1 * isarithm2*isarithm2, contourPlot);

    curveColor.w = 1.0; //-saturate(fresnel);
    gl_FragColor = curveColor;
}
