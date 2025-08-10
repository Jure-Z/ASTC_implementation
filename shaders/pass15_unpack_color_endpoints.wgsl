const BLOCK_MAX_PARTITIONS: u32 = 4u;
const BLOCK_MAX_TEXELS: u32 = 144u;
const BLOCK_MAX_WEIGHTS: u32 = 64u;

const TUNE_MAX_TRIAL_CANDIDATES = 8u;

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

struct IdealEndpointsAndWeightsPartition {
    avg: vec4<f32>,
    dir: vec4<f32>,
    endpoint0: vec4<f32>,
    endpoint1: vec4<f32>,
};


struct FinalCandidate {
    block_mode_index: u32,
    block_mode_trial_index: u32,
    total_error: f32,
    quant_level: u32, // The original quant level
    quant_level_mod: u32,

    _padding1: u32,

	color_formats_matched: u32,
    final_quant_mode: u32, // The quant mode after checking the mod version

    formats: vec4<u32>,
    quantized_weights: array<u32, BLOCK_MAX_WEIGHTS>,
    candidate_partitions: array<IdealEndpointsAndWeightsPartition, 4>,

	final_formats: vec4<u32>, //Formats can change after quantization
    packed_color_values: array<u32, 32>, //8 integers per partition
};

struct UnpackedEndpoints {
    endpoint0: array<vec4<i32>, 4>,
    endpoint1: array<vec4<i32>, 4>,
}

//--------------------------------------------------------------------------------------------------------


@group(0) @binding(0) var<uniform> uniforms: UniformVariables;
@group(0) @binding(1) var<storage, read> final_candidates: array<FinalCandidate>;

@group(0) @binding(2) var<storage, read_write> unpacked_endpoints: array<UnpackedEndpoints>;

//--------------------------------------------------------------------------------------------------------

fn uncontract_color(input: vec4<i32>) -> vec4<i32> {
	let mask = vec4<bool>(true, true, false, false);
	let bc0 = (input + input.b) >> vec4<u32>(1);
	return select(input, bc0, mask);
}

fn bit_transfer_signed(input0: ptr<function, vec4<i32>>, input1: ptr<function, vec4<i32>>) {
    var input0_val = *input0;
    var input1_val = *input1;

    //preform shifts on unsigned interegers to guarantee logical shifts
    let input0_val_u = bitcast<vec4<u32>>(input0_val);
    let input1_val_u = bitcast<vec4<u32>>(input1_val);

    input1_val =  bitcast<vec4<i32>>((input1_val_u >> vec4<u32>(1)) | (input0_val_u & vec4<u32>(0x80)));
    input0_val =  bitcast<vec4<i32>>((input0_val_u >> vec4<u32>(1)) & vec4<u32>(0x3F));

    let mask = (input0_val & vec4<i32>(0x20)) != vec4<i32>(0);
    input0_val = select(input0_val, input0_val - 0x40, mask);

    *input0 = input0_val;
    *input1 = input1_val;
}

fn luminance_unpack (
    input: array<u32, 8>,
    output0: ptr<function, vec4<i32>>,
    output1: ptr<function, vec4<i32>>
) {
    let lum0 = i32(input[0]);
    let lum1 = i32(input[1]);
    (*output0) = vec4<i32>(lum0, lum0, lum0, 255);
	(*output1) = vec4<i32>(lum1, lum1, lum1, 255);
}

fn luminance_delta_unpack (
    input: array<u32, 8>,
    output0: ptr<function, vec4<i32>>,
    output1: ptr<function, vec4<i32>>
) {
    let v0 = i32(input[0]);
    let v1 = i32(input[1]);
    
    let lum0 = (v0 >> 2) | (v1 & 0xC0);
    var lum1 = lum0 + (v1 & 0x3F);

    lum1 = min(lum1, 255);

    (*output0) = vec4<i32>(lum0, lum0, lum0, 255);
	(*output1) = vec4<i32>(lum1, lum1, lum1, 255);
}

fn luminance_alpha_unpack (
    input: array<u32, 8>,
    output0: ptr<function, vec4<i32>>,
    output1: ptr<function, vec4<i32>>
) {
    let lum0 = i32(input[0]);
    let lum1 = i32(input[1]);
    let alpha0 = i32(input[2]);
    let alpha1 = i32(input[3]);

	(*output0) = vec4<i32>(lum0, lum0, lum0, alpha0);
    (*output1) = vec4<i32>(lum1, lum1, lum1, alpha1);
}

fn luminance_alpha_delta_unpack (
    input: array<u32, 8>,
    output0: ptr<function, vec4<i32>>,
    output1: ptr<function, vec4<i32>>
) {
    var lum0 = i32(input[0]);
    var lum1 = i32(input[1]);
    var alpha0 = i32(input[2]);
    var alpha1 = i32(input[3]);

    lum0 = lum0 | ((lum1 & 0x80) << 1);
    alpha0 = alpha0 | ((alpha1 & 0x80) << 1);
    lum1 = lum1 & 0x7F;
    alpha1 = alpha1 & 0x7F;

    if((lum1 & 0x40) != 0) {
        lum1 = lum1 - 0x80;
    }

    if((alpha1 & 0x40) != 0) {
		alpha1 = alpha1 - 0x80;
    }

    lum0 = lum0 >> 1;
    lum1 = lum1 >> 1;
    alpha0 = alpha0 >> 1;
    alpha1 = alpha1 >> 1;

    lum1 = lum0 + lum1;
    alpha1 = alpha0 + alpha1;

    lum1 = clamp(lum1, 0, 255);
    alpha1 = clamp(alpha1, 0, 255);

	(*output0) = vec4<i32>(lum0, lum0, lum0, alpha0);
    (*output1) = vec4<i32>(lum1, lum1, lum1, alpha1);
}

fn rgb_scale_unpack (
    input: array<u32, 8>,
    output0: ptr<function, vec4<i32>>,
    output1: ptr<function, vec4<i32>>
) {
    var input0 = vec4<i32>(i32(input[0]), i32(input[1]), i32(input[2]), 255);
    let scale = i32(input[3]);

    (*output1) = input0;

    input0 = (input0 * scale) >> vec4<u32>(8);
    input0.a = 255;
    
    (*output0) = input0;
}

fn rgb_scale_alpha_unpack (
    input: array<u32, 8>,
    output0: ptr<function, vec4<i32>>,
    output1: ptr<function, vec4<i32>>
) {
    var input0 = vec4<i32>(i32(input[0]), i32(input[1]), i32(input[2]), i32(input[4]));
    let scale = i32(input[3]);
    let alpha0 = i32(input[4]);
    let alpha1 = i32(input[5]);

    (*output1) = vec4<i32>(input0.r, input0.g, input0.b, alpha1);

    input0 = (input0 * scale) >> vec4<u32>(8);
	input0.a = alpha0;

	(*output0) = input0;
}

fn rgba_unpack (
    input0: vec4<i32>,
    input1: vec4<i32>,
    output0: ptr<function, vec4<i32>>,
    output1: ptr<function, vec4<i32>>
) {
    var i0 = input0;
    var i1 = input1;

    //Apply blue uncontaction if needed
    if((i0.r + i0.g + i0.b) > (i1.r + i1.g + i1.b)) {
        i0 = uncontract_color(i0);
        i1 = uncontract_color(i1);

        let temp = i0;
        i0 = i1;
        i1 = temp;
    }

	(*output0) = i0;
    (*output1) = i1;
}

fn rgb_unpack (
    input0: vec4<i32>,
    input1: vec4<i32>,
    output0: ptr<function, vec4<i32>>,
    output1: ptr<function, vec4<i32>>
) {
    rgba_unpack(input0, input1, output0, output1);
	(*output0).a = 255;
	(*output1).a = 255;
}

fn rgba_delta_unpack (
    input0: vec4<i32>,
    input1: vec4<i32>,
    output0: ptr<function, vec4<i32>>,
    output1: ptr<function, vec4<i32>>
) {
    var i0 = input0;
    var i1 = input1;

    bit_transfer_signed(&i1, &i0);

    //Apply blue contraction in needed
    let rgb_sum = i1.r + i1.g + i1.b;
    i1 = i1 + i0;
    if(rgb_sum < 0) {
        i0 = uncontract_color(i0);
        i1 = uncontract_color(i1);

        let temp = i0;
        i0 = i1;
        i1 = temp;
    }

    (*output0) = clamp(i0, vec4<i32>(0), vec4<i32>(255));
    (*output1) = clamp(i1, vec4<i32>(0), vec4<i32>(255));
}

fn rgb_delta_unpack (
    input0: vec4<i32>,
    input1: vec4<i32>,
    output0: ptr<function, vec4<i32>>,
    output1: ptr<function, vec4<i32>>
) {
    rgba_delta_unpack(input0, input1, output0, output1);
    (*output0).a = 255;
    (*output1).a = 255;
}

//--------------------------------------------------------------------------------------------------------

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {

    let candidate_idx = global_id.x;
    let partition_count = uniforms.partition_count;

    let foramts = final_candidates[candidate_idx].final_formats;
    let packed_color_values = final_candidates[candidate_idx].packed_color_values;

    // Unpack the endpoints for each partition
    for(var p = 0u; p < partition_count; p = p + 1u) {
        let format = foramts[p];

        var packed_endpoint: array<u32, 8>;
        for (var i = 0u; i < 8u; i = i + 1u) {
            packed_endpoint[i] = packed_color_values[p * 8u + i];
        }

        var output0: vec4<i32>;
        var output1: vec4<i32>;

        switch (format) {
            case FMT_LUMINANCE: {
                luminance_unpack(packed_endpoint, &output0, &output1);
                break;
            }
            case FMT_LUMINANCE_DELTA: {
                luminance_delta_unpack(packed_endpoint, &output0, &output1);
                break;
            }
            case FMT_LUMINANCE_ALPHA: {
                luminance_alpha_unpack(packed_endpoint, &output0, &output1);
                break;
            }
            case FMT_LUMINANCE_ALPHA_DELTA: {
                luminance_alpha_delta_unpack(packed_endpoint, &output0, &output1);
                break;
            }
            case FMT_RGB_SCALE: {
                rgb_scale_unpack(packed_endpoint, &output0, &output1);
                break;
            }
            case FMT_RGB_SCALE_ALPHA: {
                rgb_scale_alpha_unpack(packed_endpoint, &output0, &output1);
                break;
            }
            case FMT_RGB: {
                let input0 = vec4<i32>(i32(packed_endpoint[0]), i32(packed_endpoint[2]), i32(packed_endpoint[4]), 0);
                let input1 = vec4<i32>(i32(packed_endpoint[1]), i32(packed_endpoint[3]), i32(packed_endpoint[5]), 0);
                rgb_unpack(input0, input1, &output0, &output1);
                break;
            }
            case FMT_RGB_DELTA: {
                let input0 = vec4<i32>(i32(packed_endpoint[0]), i32(packed_endpoint[2]), i32(packed_endpoint[4]), 0);
                let input1 = vec4<i32>(i32(packed_endpoint[1]), i32(packed_endpoint[3]), i32(packed_endpoint[5]), 0);
                rgb_delta_unpack(input0, input1, &output0, &output1);
                break;
            }
            case FMT_RGBA: {
                let input0 = vec4<i32>(i32(packed_endpoint[0]), i32(packed_endpoint[2]), i32(packed_endpoint[4]), i32(packed_endpoint[6]));
                let input1 = vec4<i32>(i32(packed_endpoint[1]), i32(packed_endpoint[3]), i32(packed_endpoint[5]), i32(packed_endpoint[7]));
                rgba_unpack(input0, input1, &output0, &output1);
                break;
            }
            case FMT_RGBA_DELTA: {
                let input0 = vec4<i32>(i32(packed_endpoint[0]), i32(packed_endpoint[2]), i32(packed_endpoint[4]), i32(packed_endpoint[6]));
                let input1 = vec4<i32>(i32(packed_endpoint[1]), i32(packed_endpoint[3]), i32(packed_endpoint[5]), i32(packed_endpoint[7]));
                rgba_delta_unpack(input0, input1, &output0, &output1);
                break;
            }
            default: {
                luminance_unpack(packed_endpoint, &output0, &output1);
            }
        }

        output0 = output0 * 257;
        output1 = output1 * 257;

        //store outputs
        unpacked_endpoints[candidate_idx].endpoint0[p] = output0;
        unpacked_endpoints[candidate_idx].endpoint1[p] = output1;
    }

}
