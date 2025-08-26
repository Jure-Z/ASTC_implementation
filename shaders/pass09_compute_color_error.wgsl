const WORKGROUP_SIZE: u32 = 32u;
const BLOCK_MAX_PARTITIONS: u32 = 4u;
const BLOCK_MAX_TEXELS: u32 = 144u;

const NUM_QUANT_LEVELS = 21u;
const NUM_INT_COUNTS = 4u; //8,6,4,2
const ERROR_CALC_DEFAULT = 1e37f;

//error that is generally expected for quantization level
const BASELINE_QUANT_ERROR = array<f32, 17> ( //size is 21 - 4 (QUANT_6)
	(65536.0f * 65536.0f / 18.0f) / (5 * 5),
	(65536.0f * 65536.0f / 18.0f) / (7 * 7),
	(65536.0f * 65536.0f / 18.0f) / (9 * 9),
	(65536.0f * 65536.0f / 18.0f) / (11 * 11),
	(65536.0f * 65536.0f / 18.0f) / (15 * 15),
	(65536.0f * 65536.0f / 18.0f) / (19 * 19),
	(65536.0f * 65536.0f / 18.0f) / (23 * 23),
	(65536.0f * 65536.0f / 18.0f) / (31 * 31),
	(65536.0f * 65536.0f / 18.0f) / (39 * 39),
	(65536.0f * 65536.0f / 18.0f) / (47 * 47),
	(65536.0f * 65536.0f / 18.0f) / (63 * 63),
	(65536.0f * 65536.0f / 18.0f) / (79 * 79),
	(65536.0f * 65536.0f / 18.0f) / (95 * 95),
	(65536.0f * 65536.0f / 18.0f) / (127 * 127),
	(65536.0f * 65536.0f / 18.0f) / (159 * 159),
	(65536.0f * 65536.0f / 18.0f) / (191 * 191),
	(65536.0f * 65536.0f / 18.0f) / (255 * 255),
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

    valid_decimation_mode_count: u32,
	valid_block_mode_count: u32,

    quant_limit : u32,
    partition_count : u32,
    tune_candidate_limit : u32,

    _padding1: u32,
    _padding2: u32,

    channel_weights : vec4<f32>,
};

struct InputBlock {
    pixels: array<vec4<f32>, BLOCK_MAX_TEXELS>,
    texel_partitions: array<u32, BLOCK_MAX_TEXELS>,
    partition_pixel_counts: array<u32, 4>,

    partitioning_idx: u32,
    grayscale: u32,
    constant_alpha: u32,
    padding: u32,
};

struct IdealEndpointsAndWeightsPartition {
    avg: vec4<f32>,
    dir: vec4<f32>,
    endpoint0: vec4<f32>,
    endpoint1: vec4<f32>,
};

struct IdealEndpointsAndWeights {
    partitions: array<IdealEndpointsAndWeightsPartition, 4>,
    weights: array<f32, BLOCK_MAX_TEXELS>,

    weight_error_scale: array<f32, BLOCK_MAX_TEXELS>,

    is_constant_weight_error_scale : u32,
    min_weight_cuttof : f32,
    _padding1 : u32,
    _padding2 : u32,
};

struct EncodingChoiceErrors {
    rgb_scale_error: f32,
    rgb_luma_error: f32,
    luminance_error: f32,
    alpha_drop_error: f32,

    can_offset_encode: u32,
    can_blue_contract: u32,

    _padding1: u32,
    _padding2: u32,
}


@group(0) @binding(0) var<uniform> uniforms: UniformVariables;
@group(0) @binding(1) var<storage, read> inputBlocks: array<InputBlock>;
@group(0) @binding(2) var<storage, read> ideal_endpoints_and_weights: array<IdealEndpointsAndWeights>;
@group(0) @binding(3) var<storage, read> encoding_choice_errors: array<EncodingChoiceErrors>;

@group(0) @binding(4) var<storage, read_write> output_best_error: array<f32>;
@group(0) @binding(5) var<storage, read_write> output_format_of_choice: array<u32>;



// Pre-calculated values, one for each partition in the block.
var<workgroup> part_rgb_range_error: array<f32, 4>;
var<workgroup> part_alpha_range_error: array<f32, 4>;
var<workgroup> part_base_quant_error_rgb: array<f32, 4>;
var<workgroup> part_base_quant_error_a: array<f32, 4>;
var<workgroup> part_error_scale_bc_rgba: array<f32, 4>;
var<workgroup> part_error_scale_oe_rgba: array<f32, 4>;
var<workgroup> part_error_scale_bc_rgb: array<f32, 4>;
var<workgroup> part_error_scale_oe_rgb: array<f32, 4>;


@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(workgroup_id) group_id: vec3<u32>, @builtin(local_invocation_index) local_idx: u32) {
    let block_idx = group_id.x;
    let partition_count = uniforms.partition_count;

    //precomputation
    if(local_idx < partition_count) {
        let p = local_idx;
        let part_global_idx = block_idx * uniforms.partition_count + p;

        let ep0 = ideal_endpoints_and_weights[block_idx].partitions[p].endpoint0;
        let ep1 = ideal_endpoints_and_weights[block_idx].partitions[p].endpoint1;

        let eci = encoding_choice_errors[part_global_idx];
        let partition_size = f32(inputBlocks[block_idx].partition_pixel_counts[p]);

        //Calculate range error (endpoints going out of [0, 65535] range)
        let offset = vec4<f32>(65535.0);
        let ep0_err_high = max(ep0 - offset, vec4<f32>(0.0));
        let ep1_err_high = max(ep1 - offset, vec4<f32>(0.0));
        let ep0_err_low = min(ep0, vec4<f32>(0.0));
        let ep1_err_low = min(ep1, vec4<f32>(0.0));

        let sum_range_err = (ep0_err_low * ep0_err_low) + (ep1_err_low * ep1_err_low) + (ep0_err_high * ep0_err_high) + (ep1_err_high * ep1_err_high);
        
        part_rgb_range_error[p] = dot(sum_range_err.xyz, uniforms.channel_weights.xyz) * 0.5 * partition_size;
        part_alpha_range_error[p] = sum_range_err.w * uniforms.channel_weights.w * 0.5 * partition_size;

        //precompute quantization errors
        let error_weight = uniforms.channel_weights;
        part_base_quant_error_rgb[p] = (error_weight.x + error_weight.y + error_weight.z) * partition_size;
        part_base_quant_error_a[p] = error_weight.w * partition_size;

        //precompute error scales for offset encoding and blue channel contract
        part_error_scale_bc_rgba[p] = select(1.0, 0.625, eci.can_blue_contract == 1u);
        part_error_scale_oe_rgba[p] = select(1.0, 0.5, eci.can_offset_encode == 1u);
        part_error_scale_bc_rgb[p] = select(1.0, 0.5, eci.can_blue_contract == 1u);
        part_error_scale_oe_rgb[p] = select(1.0, 0.25, eci.can_offset_encode == 1u);


        //initialize low precision quantizazion modes (QUANT_2, QUANT_3, QUANT_4, QUANT_5)
        for(var i = 0u; i < 4; i += 1u) {
            let base_index = (part_global_idx * NUM_QUANT_LEVELS + i) * NUM_INT_COUNTS;

            output_best_error[base_index + 3u] = ERROR_CALC_DEFAULT;
            output_best_error[base_index + 2u] = ERROR_CALC_DEFAULT;
            output_best_error[base_index + 1u] = ERROR_CALC_DEFAULT;
            output_best_error[base_index + 0u] = ERROR_CALC_DEFAULT;

            output_format_of_choice[base_index + 3u] = FMT_RGBA; //8 integers
            output_format_of_choice[base_index + 2u] = FMT_RGB; //6 integers
            output_format_of_choice[base_index + 1u] = FMT_RGB_SCALE; //4 integers
            output_format_of_choice[base_index + 0u] = FMT_LUMINANCE; //2 integers
        }
    }

    workgroupBarrier();

    //Pick the best endpoint encoding for every quantization level, for every integer count
    //from QUANT_6 to QUANT_256
    for(var i = local_idx; i < 17; i += WORKGROUP_SIZE) {
        let quant_idx = i + 4u; //start at QUANT_6

        for(var p = 0u; p < partition_count; p += 1u) {
			let part_global_idx = block_idx * uniforms.partition_count + p;
            let eci = encoding_choice_errors[part_global_idx];

            //get precomputed values from shared memory
            let rgb_range_err = part_rgb_range_error[p];
            let alpha_range_err = part_alpha_range_error[p];
            let base_err_rgb = part_base_quant_error_rgb[p];
            let base_err_a = part_base_quant_error_a[p];
			
            var oe_rgba = part_error_scale_oe_rgba[p];
            var oe_rgb = part_error_scale_oe_rgb[p];
            if (quant_idx >= 19) { // QUANT_192 and up
                oe_rgba = 1.0;
                oe_rgb = 1.0;
            }

            let base_quant_err = BASELINE_QUANT_ERROR[i];
            let quant_err_rgb = base_err_rgb * base_quant_err;
            let quant_err_rgba = (base_err_rgb + base_err_a) * base_quant_err;


            var best_err: array<f32, NUM_INT_COUNTS>;
            var format_choice: array<u32, NUM_INT_COUNTS>;

            //8 integers (RGBA)
            best_err[3] = quant_err_rgba * part_error_scale_bc_rgba[p] * oe_rgba + rgb_range_err + alpha_range_err;
            format_choice[3] = FMT_RGBA;

            //6 integers (RGB vs RGBS+A)
            let full_ldr_rgb_err = quant_err_rgb * part_error_scale_bc_rgb[p] * oe_rgb + rgb_range_err + eci.alpha_drop_error;
            let rgbs_alpha_err = quant_err_rgba + eci.rgb_scale_error + rgb_range_err + alpha_range_err;
            if (rgbs_alpha_err < full_ldr_rgb_err) {
                best_err[2] = rgbs_alpha_err;
                format_choice[2] = FMT_RGB_SCALE_ALPHA;
            } else {
                best_err[2] = full_ldr_rgb_err;
                format_choice[2] = FMT_RGB;
            }

            // 4 integers (RGBS vs LA+LA)
            let ldr_rgbs_err = quant_err_rgb + rgb_range_err + eci.alpha_drop_error + eci.rgb_scale_error;
            let lum_alpha_err = quant_err_rgba + rgb_range_err + alpha_range_err + eci.luminance_error;
            if (ldr_rgbs_err < lum_alpha_err) {
                best_err[1] = ldr_rgbs_err;
                format_choice[1] = FMT_RGB_SCALE;
            } else {
                best_err[1] = lum_alpha_err;
                format_choice[1] = FMT_LUMINANCE_ALPHA;
            }

            // 2 integers (Luminance)
            best_err[0] = quant_err_rgb + rgb_range_err + eci.alpha_drop_error + eci.luminance_error;
            format_choice[0] = FMT_LUMINANCE;


            //write results for this partition & quantization level
            let base_index = (part_global_idx * NUM_QUANT_LEVELS + quant_idx) * NUM_INT_COUNTS;
            for (var j = 0u; j < 4u; j = j + 1u) {
                output_best_error[base_index + j] = best_err[j];
                output_format_of_choice[base_index + j] = format_choice[j];
            }
        }
    }

}