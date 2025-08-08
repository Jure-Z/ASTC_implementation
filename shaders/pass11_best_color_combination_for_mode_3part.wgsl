const WORKGROUP_SIZE: u32 = 64u;
const BLOCK_MAX_WEIGHTS: u32 = 64u;
const NUM_QUANT_LEVELS: u32 = 21u;
const NUM_COMBINED_INT_COUNTS: u32 = 10u;
const MAX_BITS: u32 = 128u;
const ERROR_CALC_DEFAULT: f32 = 1e37;

//precomputed quantization levels for integer count and avalible bits
//-1 if integer count cannot fit in the available bits
//the table is flattened, indexed by: integer_count * MAX_BITS + bit_count
const QUANT_MODE_TABLE = array<i32, 1280>(
    //0 integers
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    //2 integers
    -1, -1,  0,  0,  2,  3,  5,  6,  8,  9, 11, 12, 14, 15, 17, 18, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    //4 integers
    -1, -1, -1, -1,  0,  0,  0,  1,  2,  2,  3,  4,  5,  5,  6,  7,  8,  8,  9, 10, 11, 11, 12, 13, 14, 14, 15, 16, 17, 17, 18, 19,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    //6 integers
    -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,  7,  8,  8,  9,  9, 10, 10, 11, 11,
    12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    //8 integers
    -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  4,  4,  4,  5,  5,  5,  6,  6,  7,  7,  7,
     8,  8,  8,  9,  9, 10, 10, 10, 11, 11, 11, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 19, 19, 19,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    //10 integers
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  4,  4,  4,  4,  5,  5,
     5,  5,  6,  6,  7,  7,  7,  7,  8,  8,  8,  8,  9,  9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14,
    15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    //12 integers
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3,
     4,  4,  4,  4,  5,  5,  5,  5,  6,  6,  6,  6,  7,  7,  7,  7,  8,  8,  8,  8,  9,  9,  9,  9, 10, 10, 10, 10, 11, 11, 11, 11,
    12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    //14 integers
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  2,  2,  2,  2,
     2,  3,  3,  3,  3,  4,  4,  4,  4,  4,  5,  5,  5,  5,  5,  6,  6,  6,  6,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  9,  9,  9,
     9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16,
    16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    //16 integers
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,
     2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,
    8,  8,  8,  8,  8,  8,  9,  9,  9,  9, 10, 10, 10, 10, 10, 10,  11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13,
    14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19,
    //18 integers
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,
     1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,
     6,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11,
    12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 17, 17
);

//ASTC endpoint formats
const FMT_LUMINANCE = 0u;
const FMT_LUMINANCE_DELTA = 1u;
const FMT_HDR_LUMINANCE_LARGE_RANGE = 2u;
const FMT_HDR_LUMINANCE_SMALL_RANGE = 3u;
const FMT_LUMINANCE_ALPHA = 4u;
const FMT_LUMINANCE_ALPHA_DELTA = 5u;
const FMT_RGB_SCALE = 6u;
const FMT_HDR_RGB_SCALE = 7u;
const FMT_RGB = 8u;
const FMT_RGB_DELTA = 9u;
const FMT_RGB_SCALE_ALPHA = 10u;
const FMT_HDR_RGB = 11u;
const FMT_RGBA = 12u;
const FMT_RGBA_DELTA = 13u;
const FMT_HDR_RGB_LDR_ALPHA = 14u;
const FMT_HDR_RGBA = 15u;




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

struct BlockModeTrial {
    block_index : u32,
    block_mode_index : u32,
    decimation_mode_trial_index : u32,

    _padding1 : u32,
};

struct QuantizationResult {
    error: f32,
    bitcount: i32,

    _padding1: u32,
    _padding2: u32,

    quantized_weights: array<u32, BLOCK_MAX_WEIGHTS>,
};

struct CombinedEndpointFormats {
	error: f32,

    _padding0: u32,
    _padding1: u32,
    _padding2: u32,

    formats: vec4<u32>,
};

struct ColorCombinationResult {
    total_error: f32,
    best_quant_level: u32,
    best_quant_level_mod: u32,

    _padding1: u32,

    best_ep_formats: vec4<u32>,
};


@group(0) @binding(0) var<uniform> uniforms: UniformVariables;
@group(0) @binding(1) var<storage, read> block_mode_trials: array<BlockModeTrial>;
@group(0) @binding(2) var<storage, read> quantization_results: array<QuantizationResult>;
@group(0) @binding(3) var<storage, read> combined_endpoint_formats: array<CombinedEndpointFormats>;

@group(0) @binding(4) var<storage, read_write> output_color_combination_results: array<ColorCombinationResult>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let trial_idx = global_id.x;

    let quant_result = quantization_results[trial_idx];
    let weight_error = quant_result.error;
    let bits_avalible = quant_result.bitcount;

    let block_idx = block_mode_trials[trial_idx].block_index;


    //Skip if error is already to high
    if(weight_error >= ERROR_CALC_DEFAULT) {
        let out_ptr = &output_color_combination_results[trial_idx];
        (*out_ptr).total_error = ERROR_CALC_DEFAULT;
        return;
    }

    var best_total_int_count_idx = 0u;
    var best_color_error = ERROR_CALC_DEFAULT;

    let combined_error_base = block_idx * NUM_QUANT_LEVELS * NUM_COMBINED_INT_COUNTS;

    for (var int_count_idx = 3u; int_count_idx <= 9u; int_count_idx = int_count_idx + 1u) {

        let quant_table_idx = (int_count_idx * MAX_BITS) + u32(bits_avalible);
        let quant_level = QUANT_MODE_TABLE[quant_table_idx];

        //We don't have enough bits to represent a given endpoint format
        if(quant_level < 4) { //QUANT_6 = 4
			break;
		}

        let combined_error_idx = combined_error_base + u32(quant_level) * NUM_COMBINED_INT_COUNTS + int_count_idx - 3u;
        let combined_error = combined_endpoint_formats[combined_error_idx].error;
        if (combined_error < best_color_error) {
            best_color_error = combined_error;
            best_total_int_count_idx = int_count_idx;
        }
    }

    let final_quant_level = QUANT_MODE_TABLE[(best_total_int_count_idx * MAX_BITS) + u32(bits_avalible)];
    let final_quant_level_mod = QUANT_MODE_TABLE[(best_total_int_count_idx * MAX_BITS) + u32(bits_avalible) + 5u];

    var final_best_formats = vec4<u32>(0u); //default formats
    if (final_quant_level >= 4) { //QUANT_6 = 4
		let combined_error_idx = combined_error_base + u32(final_quant_level) * NUM_COMBINED_INT_COUNTS + best_total_int_count_idx - 3u;
		final_best_formats = combined_endpoint_formats[combined_error_idx].formats;
	}

    //Finalization
    let total_error = best_color_error + weight_error;

    let out_ptr = &output_color_combination_results[trial_idx];
    (*out_ptr).total_error = total_error;
    (*out_ptr).best_quant_level = u32(final_quant_level);
    (*out_ptr).best_quant_level_mod = u32(final_quant_level_mod);
    (*out_ptr).best_ep_formats = final_best_formats;
}