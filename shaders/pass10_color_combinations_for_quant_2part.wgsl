const WORKGROUP_SIZE: u32 = 32u;
const BLOCK_MAX_PARTITIONS: u32 = 4u;
const NUM_QUANT_LEVELS: u32 = 21u;
const NUM_INT_COUNTS: u32 = 4u;  // 2, 4, 6, 8 integers
const NUM_COMBINED_INT_COUNTS: u32 = 7u; // Total integers can be 2+2=4 up to 8+8=16. Indices 0..6 for i+j.
const ERROR_CALC_DEFAULT: f32 = 1e37;

struct UniformVariables {
    xdim : u32,
    ydim : u32,

    texel_count : u32,

    decimation_mode_count : u32,
    block_mode_count : u32,

    quant_limit : u32,
    partition_count : u32,
    tune_candidate_limit : u32,

    channel_weights : vec4<f32>,
};

struct CombinedEndpointFormats {
	error: f32,

    _padding0: u32,
    _padding1: u32,
    _padding2: u32,

    formats: vec4<u32>,
};


@group(0) @binding(0) var<uniform> uniforms: UniformVariables;
@group(0) @binding(1) var<storage, read> color_error_table: array<f32>;
@group(0) @binding(2) var<storage, read> format_choice_table: array<u32>;

@group(0) @binding(3) var<storage, read_write> combined_endpoint_formats: array<CombinedEndpointFormats>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(workgroup_id) group_id: vec3<u32>, @builtin(local_invocation_index) local_idx: u32) {
    let block_idx = group_id.x;

    //Initialize output buffer to highest possible error
    let total_slots_to_init = NUM_QUANT_LEVELS * NUM_COMBINED_INT_COUNTS;
    let output_base_idx = block_idx * NUM_QUANT_LEVELS * NUM_COMBINED_INT_COUNTS;

    for(var i = local_idx; i < total_slots_to_init; i += WORKGROUP_SIZE) {
        let out_ptr = &combined_endpoint_formats[output_base_idx + i];
        (*out_ptr).error = ERROR_CALC_DEFAULT;
    }

    workgroupBarrier();

    //For every quant level, find the best color format combinatoin for every integer count
    let p0_error_base = (block_idx * BLOCK_MAX_PARTITIONS + 0u) * NUM_QUANT_LEVELS * NUM_INT_COUNTS;
    let p1_error_base = (block_idx * BLOCK_MAX_PARTITIONS + 1u) * NUM_QUANT_LEVELS * NUM_INT_COUNTS;

    if(local_idx < (NUM_QUANT_LEVELS - 4)) { //QUANT_6 = 4
        let quant_level = local_idx + 4u; //Start from QUANT_6

        // Loop through the 4 integer count choices for partition 0
        for (var i = 0u; i < 4u; i = i + 1u) {
            // Loop through the 4 integer count choices for partition 1
            for (var j = 0u; j < 4u; j = j + 1u) {

                //Number of integers used for each partition can only differ by one step
                let low2 = min(i, j);
                let high2 = max(i, j);
                if (high2 - low2 > 1u) {
					continue; // Skip if the difference is more than 1
				}

                let total_int_count = i + j;

                // Read the pre-computed errors for this pairing.
                let error0 = color_error_table[p0_error_base + quant_level * NUM_INT_COUNTS + i];
                let error1 = color_error_table[p1_error_base + quant_level * NUM_INT_COUNTS + j];
                let total_error = min(error0 + error1, 1e10); // Clamp to avoid huge values

                // Check if this pairing is the new best for this total_int_count.
                let out_idx = output_base_idx + quant_level * NUM_COMBINED_INT_COUNTS + total_int_count;
                let out_ptr = &combined_endpoint_formats[out_idx];

                if (total_error < (*out_ptr).error) {
                    let format0 = format_choice_table[p0_error_base + quant_level * NUM_INT_COUNTS + i];
                    let format1 = format_choice_table[p1_error_base + quant_level * NUM_INT_COUNTS + j];
                    (*out_ptr).error = total_error;
                    (*out_ptr).formats = vec4<u32>(format0, format1, 0, 0);
                }
            }
        }
    }
}