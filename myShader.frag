varying vec4 position;
varying vec3 normal;
varying vec4 camera_dir;
varying vec4 pos_cam_spc;

void main( void )
{  
   vec3 spectrum[7];
   spectrum[0] = vec3(1.0, 1.0, 1.0 );
   spectrum[1] = vec3(0.0, 0.0, 1.0 );
   spectrum[2] = vec3(0.0, 1.0, 1.0 );
   spectrum[3] = vec3(0.0, 1.0, 0.0 );
   spectrum[4] = vec3(1.0, 1.0, 0.0 );
   spectrum[5] = vec3(1.0, 0.0, 1.0 );
   spectrum[6] = vec3(1.0, 0.0, 0.0 );
   
   float value = position.y*7.0;
   int i = int(clamp(value, 0.0, 7.0-1.0));
   int j = int(clamp((value+1.0), 0.0, 7.0-1.0));
   float t = value-float(i);
   
   vec4 color = vec4(spectrum[i].x*(1.0-t) + spectrum[j].x*t,
                       spectrum[i].y*(1.0-t) + spectrum[j].y*t,
                       spectrum[i].z*(1.0-t) + spectrum[j].z*t, 1.0);
   value = position.y*7.0 - floor(position.y*7.0);
   value = 1.0 - value * value * value * value + 0.1;
   
   float value2 = position.y*70.0 - floor(position.y*70.0);
   value2 = 1.0 - value2 * value2 * value2 * value2 + 0.1;
   
   float lighting = dot(camera_dir, normalize(vec4(normal, 0.0)));
   //float lighting = dot(normalize(realCamera), vec4(normal, 1.0));
   //float lighting = clamp(dot(normalize(position - realCamera), vec4(normal, 1.0)), 0.0, 1.0);
   //float lighting = dot(cameraPos, vec3(1.0, 1.0, 1.0));
   float mean = (color.x + color.y + color.z)*0.333333;
   
   value2 = clamp(value2 + max(0.0 , pos_cam_spc.z - 1.0)/3.0, 0.0, 1.0);
   value = value * (0.5 + value2 * 0.5);
   
   //gl_FragColor = (color * (mean + (1.0 - mean) * lighting)) * clamp(sqrt(value), 0.0, 1.0);
   gl_FragColor = color * clamp(sqrt(value), 0.0, 1.0);
}