// GLSL fragment shader
//varying vec3 eyeSpacePos;
//varying vec3 worldSpaceNormal;
//varying vec3 eyeSpaceNormal;
varying float vertex_height;
varying float shadow;


uniform sampler2D tex;
uniform sampler2D tex_color;
uniform float colorTextureFactor;
uniform float contourPlot;
uniform float yScale;
uniform float yOffset;
uniform float logScale;
uniform vec4 fixedColor;
uniform vec4 clearColor;

float heightValue(float v) {
    float a = v == 0.0 ? 0.0 : 1.0;
    float b = 0.99/98.0 * exp(-yOffset * log(98.0)) - 0.01/98.0;
    float x1 = yScale / (log(1.0) - log(b));
    float x2 = - log(b) * x1;
    float logvalue = log(v) * x1 + x2;
    float h = mix(v*yScale + yOffset, logvalue, logScale);
    return v == 0.0 ? 0.0 : max(0.01, h);
}

void main()
{
    float v = texture2D(tex, gl_TexCoord[0].xy).x;

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
