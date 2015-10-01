// GLSL vertex shader
attribute highp vec4 qt_Vertex;
varying highp float vertex_height;
varying mediump float shadow;
varying highp vec2 texCoord;
#ifdef USE_MIPMAP
varying mediump float vbias;
#endif

uniform highp sampler2D tex;
uniform mediump float flatness;
uniform mediump float yScale;
uniform mediump float yOffset;
uniform mediump vec3 logScale;
uniform mediump vec2 scale_tex;
uniform mediump vec2 offset_tex;
uniform mediump vec2 tex_delta;
uniform highp mat4 ModelViewProjectionMatrix;
uniform highp mat4 ModelViewMatrix;
uniform highp mat4 NormalMatrix;
uniform mediump vec4 vertexTextureBiasX;
uniform mediump vec4 vertexTextureBiasY;

#ifdef DRAW3D
mediump float heightValue(mediump float v) {
//    v *= 0.01; // map from small range of float16 to float32 (TfrBlockUpdater)
    mediump float h = mix(v*yScale + yOffset,       // linear
                  log(v) * logScale.y + logScale.z, // log
                  logScale.x);                      // choose

    h *= flatness;

    return v == 0.0 ? 0.0 : max(0.0, h);
}


mediump float computeShadow(highp vec4 pos, mediump vec2 tex0, mediump vec2 vb) {
    mediump float dx = tex_delta.x*exp2(vb.x);
    mediump float dy = tex_delta.y*exp2(vb.y);
    mediump float bias = min(vb.x,vb.y);
    mediump float heightx1     = texture2DLod(tex, vec2(tex0.x-dx, tex0.y), bias).x;
    mediump float heightx2     = texture2DLod(tex, vec2(tex0.x+dx, tex0.y), bias).x;
    mediump float heighty1     = texture2DLod(tex, vec2(tex0.x, tex0.y-dy), bias).x;
    mediump float heighty2     = texture2DLod(tex, vec2(tex0.x, tex0.y+dy), bias).x;
    // Use textureGrad on GLSL version 130 or GLSL version 300 es
    heightx1 = heightValue(heightx1);
    heightx2 = heightValue(heightx2);
    heighty1 = heightValue(heighty1);
    heighty2 = heightValue(heighty2);

    highp vec4 worldSpaceNormal;
    // calculate surface normal from slope for shading
    highp vec2 slope       = vec2(heightx2-heightx1, heighty2-heighty1);
    worldSpaceNormal.xyz = cross( vec3(0.0,            slope.y, 2.0*dy),
                                   vec3(2.0*dx,  slope.x, 0.0));
//    highp vec2 slope       = vec2(height-heightx1, height-heighty1);
//    worldSpaceNormal.xyz = cross( vec3(0.0,            slope.y, tex0.y-tex1.y),
//                                   vec3(tex0.x-tex1.x,  slope.x, 0.0));
    worldSpaceNormal.w = 1.0;

    highp vec3 eyeSpacePos      = (ModelViewMatrix * pos).xyz;
    highp vec3 eyeSpaceNormal   = (NormalMatrix * worldSpaceNormal).xyz;

    eyeSpacePos      = normalize(eyeSpacePos);
    eyeSpaceNormal   = normalize(eyeSpaceNormal);
    worldSpaceNormal = normalize(worldSpaceNormal);

    highp float facing    = max(0.0, dot(eyeSpaceNormal, -eyeSpacePos));
    highp float diffuse   = max(0.0, worldSpaceNormal.y); // max(0.0, dot(worldSpaceNormalVector, lightDir));
    highp float ambient   = 0.6;

    //shadow = clamp( 0.5 + diffuse+facing + fresnel, 0.5, 1.0);
    return mix(1.0, min( ambient + (diffuse+facing)*(1.0-ambient), 1.0), flatness);
}
#endif

void main()
{
    // We want linear interpolation all the way out to the edge
    mediump vec2 tex0 = qt_Vertex.xy*scale_tex + offset_tex;

    texCoord = tex0;

    mediump vec4 bx = vertexTextureBiasX;
    mediump vec4 by = vertexTextureBiasY;
    mediump vec2 vb = vec2(mix(mix(bx.x, bx.z, tex0.x), mix(bx.y, bx.w, tex0.x), tex0.y),
                           mix(mix(by.x, by.z, tex0.x), mix(by.y, by.w, tex0.x), tex0.y));
    vb = log2(vb);
    mediump float bias = min(vb.x,vb.y);
#ifdef USE_MIPMAP
    vbias = bias;
#endif
#ifdef DRAW3D
    mediump float height = texture2DLod(tex, tex0, bias).x;
    height = heightValue(height);

    highp vec4 pos = qt_Vertex.xzyw; // swizzle
    pos.y = height; // and set vertex height from texture

#ifdef DRAWISARITHM
    vertex_height = height;
#endif

#ifndef NOSHADOW
    shadow = computeShadow(pos, tex0, vb);
#endif

    // edge dropout to eliminate visible glitches between blocks
    if (pos.x<0.0 || pos.z<0.0 || pos.x>1.0 || pos.z>1.0)
        pos.y = 0.0;
#else
    highp vec4 pos = qt_Vertex.xzyw; // swizzle
#endif

    // transform to homogeneous clip space
    gl_Position      = ModelViewProjectionMatrix * pos;
}
