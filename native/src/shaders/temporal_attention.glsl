#version 450 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Query {
    float data[];
} query_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Key {
    float data[];
} key_buffer;
layout(set = 0, binding = 3, std430) readonly buffer Value {
    float data[];
} value_buffer;

layout(push_constant) uniform Parameters {
    uint sequences;
    uint frames;
    uint channels;
    uint heads;
} parameters;

void main() {
    const uint channel = gl_GlobalInvocationID.x;
    const uint query_frame = gl_GlobalInvocationID.y;
    const uint sequence = gl_GlobalInvocationID.z;
    if (channel >= parameters.channels ||
        query_frame >= parameters.frames ||
        sequence >= parameters.sequences) {
        return;
    }
    const uint head_channels = parameters.channels / parameters.heads;
    const uint head = channel / head_channels;
    const uint head_base = head * head_channels;
    const uint query_row =
        (sequence * parameters.frames + query_frame) *
        parameters.channels;
    float maximum = -3.402823466e+38;
    float scores[32];
    const float scale = inversesqrt(float(head_channels));
    for (uint key_frame = 0;
         key_frame < parameters.frames;
         ++key_frame) {
        const uint key_row =
            (sequence * parameters.frames + key_frame) *
            parameters.channels;
        float score = 0.0;
        for (uint inner = 0; inner < head_channels; ++inner) {
            score += query_buffer.data[query_row + head_base + inner] *
                key_buffer.data[key_row + head_base + inner];
        }
        score *= scale;
        scores[key_frame] = score;
        maximum = max(maximum, score);
    }
    float denominator = 0.0;
    for (uint key_frame = 0;
         key_frame < parameters.frames;
         ++key_frame) {
        scores[key_frame] = exp(scores[key_frame] - maximum);
        denominator += scores[key_frame];
    }
    float result = 0.0;
    for (uint key_frame = 0;
         key_frame < parameters.frames;
         ++key_frame) {
        const uint key_row =
            (sequence * parameters.frames + key_frame) *
            parameters.channels;
        result += scores[key_frame] / denominator *
            value_buffer.data[key_row + channel];
    }
    output_buffer.data[query_row + channel] = result;
}
