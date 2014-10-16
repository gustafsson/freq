// GLSL vertex shader
attribute highp vec4 qt_Vertex;
varying highp float vertex_height;
varying mediump float shadow;
varying highp vec2 texCoord;

uniform highp sampler2D tex;
uniform mediump float flatness;
uniform mediump float yScale;
uniform mediump float yOffset;
uniform mediump vec3 logScale;
uniform mediump vec2 scale_tex;
uniform mediump vec2 offset_tex;
uniform highp mat4 ModelViewProjectionMatrix;
uniform highp mat4 ModelViewMatrix;
uniform highp mat4 NormalMatrix;

mediump float heightValue(mediump float v) {
    v *= 0.01; // map from small range of float16 to float32 (TfrBlockUpdater)
    mediump float h = mix(v*yScale + yOffset,       // linear
                  log(v) * logScale.y + logScale.z, // log
                  logScale.x);                      // choose

    h *= flatness;

    return v == 0.0 ? 0.0 : max(0.0, h);
}


void main()
{
    // We want linear interpolation all the way out to the edge
    mediump vec2 vertex = qt_Vertex.xz;
    mediump vec2 tex0 = vertex*scale_tex + offset_tex;

    texCoord = tex0;

    mediump vec2 tex1 = max(tex0 - offset_tex*2.0, offset_tex);
    mediump vec2 tex2 = min(tex0 + offset_tex*2.0, 1.0-offset_tex);

    mediump float height       = texture2D(tex, tex0).x;
    mediump float heightx1     = texture2D(tex, vec2(tex1.x, tex0.y)).x;
    mediump float heightx2     = texture2D(tex, vec2(tex2.x, tex0.y)).x;
    mediump float heighty1     = texture2D(tex, vec2(tex0.x, tex1.y)).x;
    mediump float heighty2     = texture2D(tex, vec2(tex0.x, tex2.y)).x;

    height = heightValue(height);
    heightx1 = heightValue(heightx1);
    heightx2 = heightValue(heightx2);
    heighty1 = heightValue(heighty1);
    heighty2 = heightValue(heighty2);

//    height = texture2DLod(tex, texCoord, 0.0).x;
    vertex_height = height;

    // edge dropout to eliminate visible glitches between blocks
    if (vertex.x<0.0 || vertex.y<0.0 || vertex.x>=1.0 || vertex.y>=1.0)
        height = 0.0;

    highp vec4 pos         = vec4(vertex.x, height, vertex.y, 1.0);

    // transform to homogeneous clip space
    gl_Position      = ModelViewProjectionMatrix * pos;

    highp vec4 worldSpaceNormal;
    // calculate surface normal from slope for shading
    highp vec2 slope       = vec2(heightx2-heightx1, heighty2-heighty1);
    worldSpaceNormal.xyz = cross( vec3(0.0,            slope.y, tex2.y-tex1.y),
                                   vec3(tex2.x-tex1.x,  slope.x, 0.0));
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

    //shadow = clamp( 0.5 + diffuse+facing + fresnel, 0.5, 1.0);
    shadow = mix(1.0, min( 0.7 + (diffuse+facing)*0.3, 1.0), flatness);
}
