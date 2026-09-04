#version 450 core

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    float data[];
} input_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Position {
    float data[];
} position_buffer;

layout(push_constant) uniform Parameters {
    uint sequences;
    uint frames;
    uint channels;
    uint position_offset;
} parameters;

void main() {
    const uint index = gl_GlobalInvocationID.x;
    const uint count =
        parameters.sequences * parameters.frames * parameters.channels;
    if (index >= count) return;
    const uint channel = index % parameters.channels;
    const uint frame =
        (index / parameters.channels) % parameters.frames;
    output_buffer.data[index] =
        input_buffer.data[index] +
        position_buffer.data[
            (parameters.position_offset + frame) *
                parameters.channels + channel];
}
