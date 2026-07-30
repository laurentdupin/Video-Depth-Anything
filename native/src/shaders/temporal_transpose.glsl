#version 450 core

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    float data[];
} input_buffer;

layout(push_constant) uniform Parameters {
    uint frames;
    uint channels;
    uint spatial;
    uint to_sequence_major;
} parameters;

void main() {
    const uint index = gl_GlobalInvocationID.x;
    const uint count =
        parameters.frames * parameters.channels * parameters.spatial;
    if (index >= count) return;
    if (parameters.to_sequence_major != 0) {
        const uint channel = index % parameters.channels;
        const uint frame_spatial = index / parameters.channels;
        const uint position = frame_spatial % parameters.spatial;
        const uint frame = frame_spatial / parameters.spatial;
        output_buffer.data[
            (position * parameters.frames + frame) *
                parameters.channels +
            channel] = input_buffer.data[index];
    } else {
        const uint channel = index % parameters.channels;
        const uint sequence_frame = index / parameters.channels;
        const uint frame = sequence_frame % parameters.frames;
        const uint position = sequence_frame / parameters.frames;
        output_buffer.data[
            (frame * parameters.spatial + position) *
                parameters.channels +
            channel] = input_buffer.data[index];
    }
}
