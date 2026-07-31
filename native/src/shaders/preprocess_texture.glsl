#version 450 core
layout(local_size_x=8, local_size_y=8, local_size_z=1) in;
layout(binding=0) uniform sampler2D source_texture;
layout(std430,binding=1) writeonly buffer Destination { float data[]; } output_data;
layout(push_constant) uniform Parameters { uint width; uint height; uint size; } p;
void main() {
    uint x=gl_GlobalInvocationID.x,y=gl_GlobalInvocationID.y;
    if(x>=p.size||y>=p.size) return;
    uint sx=min(p.width-1u,x*p.width/p.size);
    uint sy=min(p.height-1u,y*p.height/p.size);
    vec4 pixel=texelFetch(source_texture,ivec2(sx,sy),0);
    uint index=y*p.size+x, plane=p.size*p.size;
    output_data.data[index]=(pixel.b-0.485)/0.229;
    output_data.data[plane+index]=(pixel.g-0.456)/0.224;
    output_data.data[2u*plane+index]=(pixel.r-0.406)/0.225;
}
