#version 450 core

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    float data[];
} input_buffer;

layout(push_constant) uniform Parameters {
    uint sequences;
    uint frames;
    uint channels;
    uint frame;
} parameters;

void main() {
    const uint index = gl_GlobalInvocationID.x;
    const uint count = parameters.sequences * parameters.channels;
    if (index >= count) return;
    const uint sequence = index / parameters.channels;
    const uint channel = index % parameters.channels;
    output_buffer.data[index] = input_buffer.data[
        (sequence * parameters.frames + parameters.frame) *
        parameters.channels + channel];
}
