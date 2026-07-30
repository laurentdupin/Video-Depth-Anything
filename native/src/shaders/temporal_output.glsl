#version 450 core

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Projected {
    float data[];
} projected_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Residual {
    float data[];
} residual_buffer;

layout(push_constant) uniform Parameters {
    uint frames;
    uint channels;
    uint spatial;
} parameters;

void main() {
    const uint index = gl_GlobalInvocationID.x;
    const uint count =
        parameters.frames * parameters.channels * parameters.spatial;
    if (index >= count) return;
    const uint position = index % parameters.spatial;
    const uint frame_channel = index / parameters.spatial;
    const uint channel = frame_channel % parameters.channels;
    const uint frame = frame_channel / parameters.channels;
    const uint projected_index =
        (frame * parameters.spatial + position) *
            parameters.channels +
        channel;
    output_buffer.data[index] =
        projected_buffer.data[projected_index] +
        residual_buffer.data[index];
}
