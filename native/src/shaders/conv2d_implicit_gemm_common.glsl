layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    float data[];
} input_buffer;
#if defined(HALF_WEIGHT)
layout(set = 0, binding = 2, std430) readonly buffer Weight {
    uint data[];
} weight_buffer;
float read_weight(uint index) {
    const vec2 values =
        unpackHalf2x16(weight_buffer.data[index >> 1]);
    return (index & 1) == 0 ? values.x : values.y;
}
#else
layout(set = 0, binding = 2, std430) readonly buffer Weight {
    float data[];
} weight_buffer;
float read_weight(uint index) {
    return weight_buffer.data[index];
}
#endif
layout(set = 0, binding = 3, std430) readonly buffer Bias {
    float data[];
} bias_buffer;
layout(push_constant) uniform Parameters {
    uint input_width;
    uint input_height;
    uint input_channels;
    uint output_width;
    uint output_height;
    uint output_channels;
    uint kernel;
    uint stride;
    int padding;
    uint has_bias;
    uint batches;
    uint output_channel_blocks;
} parameters;

const uint inner_tile = 32;
shared float input_tile[32 * inner_tile];
shared float weight_tile[32 * inner_tile];

float read_input(uint row, uint inner) {
    const uint output_spatial =
        parameters.output_width * parameters.output_height;
    const uint batch = row / output_spatial;
    const uint spatial = row % output_spatial;
    const uint output_y = spatial / parameters.output_width;
    const uint output_x = spatial % parameters.output_width;
    const uint kernel_area = parameters.kernel * parameters.kernel;
    const uint channel = inner / kernel_area;
    const uint kernel_offset = inner % kernel_area;
    const int input_y =
        int(output_y * parameters.stride +
            kernel_offset / parameters.kernel) - parameters.padding;
    const int input_x =
        int(output_x * parameters.stride +
            kernel_offset % parameters.kernel) - parameters.padding;
    if (batch >= parameters.batches ||
        channel >= parameters.input_channels ||
        input_x < 0 || input_x >= int(parameters.input_width) ||
        input_y < 0 || input_y >= int(parameters.input_height))
        return 0.0;
    return input_buffer.data[
        ((batch * parameters.input_channels + channel) *
            parameters.input_height + uint(input_y)) *
            parameters.input_width + uint(input_x)];
}

void main() {
    const uint rows = parameters.batches *
        parameters.output_width * parameters.output_height;
    const uint inner_count =
        parameters.input_channels * parameters.kernel * parameters.kernel;
    const uint row_base =
        gl_WorkGroupID.x * 32 + gl_LocalInvocationID.x * 4;
    const uint output_channel_base =
        gl_WorkGroupID.y * 32 + gl_LocalInvocationID.y * 4;
    const uint lane =
        gl_LocalInvocationID.y * gl_WorkGroupSize.x +
        gl_LocalInvocationID.x;
    float sums[4][4];
    for (uint output_offset = 0; output_offset < 4; ++output_offset)
        for (uint row_offset = 0; row_offset < 4; ++row_offset)
            sums[output_offset][row_offset] = 0.0;
    for (uint inner_base = 0;
         inner_base < inner_count;
         inner_base += inner_tile) {
        for (uint index = lane; index < 32 * inner_tile; index += 64) {
            const uint tile_row = index / inner_tile;
            const uint inner = inner_base + index % inner_tile;
            const uint row = gl_WorkGroupID.x * 32 + tile_row;
            input_tile[index] =
                row < rows && inner < inner_count
                ? read_input(row, inner) : 0.0;
        }
        for (uint index = lane; index < 32 * inner_tile; index += 64) {
            const uint tile_output = index / inner_tile;
            const uint inner = inner_base + index % inner_tile;
            const uint output_channel =
                gl_WorkGroupID.y * 32 + tile_output;
            weight_tile[index] =
                output_channel < parameters.output_channels &&
                inner < inner_count
                ? read_weight(output_channel * inner_count + inner)
                : 0.0;
        }
        barrier();
        const uint tile_inner_count =
            min(inner_tile, inner_count - inner_base);
        for (uint inner_offset = 0;
             inner_offset < tile_inner_count;
             ++inner_offset) {
            float input_values[4];
            float weight_values[4];
            for (uint row_offset = 0; row_offset < 4; ++row_offset)
                input_values[row_offset] = input_tile[
                    (gl_LocalInvocationID.x * 4 + row_offset) *
                        inner_tile + inner_offset];
            for (uint output_offset = 0; output_offset < 4;
                 ++output_offset)
                weight_values[output_offset] = weight_tile[
                    (gl_LocalInvocationID.y * 4 + output_offset) *
                        inner_tile + inner_offset];
            for (uint output_offset = 0; output_offset < 4;
                 ++output_offset)
                for (uint row_offset = 0; row_offset < 4; ++row_offset)
                    sums[output_offset][row_offset] +=
                        weight_values[output_offset] *
                        input_values[row_offset];
        }
        barrier();
    }
    const uint output_spatial =
        parameters.output_width * parameters.output_height;
    for (uint output_offset = 0; output_offset < 4; ++output_offset) {
        const uint output_channel =
            output_channel_base + output_offset;
        if (output_channel >= parameters.output_channels) continue;
        const float bias = parameters.has_bias != 0
            ? bias_buffer.data[output_channel] : 0.0;
        for (uint row_offset = 0; row_offset < 4; ++row_offset) {
            const uint row = row_base + row_offset;
            if (row >= rows) continue;
            const uint batch = row / output_spatial;
            const uint spatial = row % output_spatial;
            output_buffer.data[
                (batch * parameters.output_channels + output_channel) *
                    output_spatial + spatial] =
                sums[output_offset][row_offset] + bias;
        }
    }
}
