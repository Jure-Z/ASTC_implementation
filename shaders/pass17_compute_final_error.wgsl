const WORKGROUP_SIZE: u32 = 64u;
const BLOCK_MAX_TEXELS: u32 = 144u;
const BLOCK_MAX_WEIGHTS: u32 = 64u;
const BLOCK_MAX_PARTITIONS: u32 = 4u;
const ERROR_CALC_DEFAULT: f32 = 1e37;



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

struct BlockMode {
	mode_index : u32,
    decimation_mode : u32,
    quant_mode : u32,
    weight_bits : u32,
    is_dual_plane : u32,

    _padding1 : u32,
    _padding2 : u32,
    _padding3 : u32,
};

struct DecimationInfo {
    texel_count : u32,
    weight_count : u32,
    weight_x : u32,
    weight_y : u32,

    max_quant_level : u32,
    max_angular_steps : u32,
    max_quant_steps: u32,
    _padding: u32,

    texel_weight_count : array<u32, BLOCK_MAX_TEXELS>,
    texel_weights_offset : array<u32, BLOCK_MAX_TEXELS>,

    weight_texel_count : array<u32, BLOCK_MAX_WEIGHTS>,
    weight_texels_offset : array<u32, BLOCK_MAX_WEIGHTS>,
};

struct TexelToWeightMap {
	weight_index : u32,
	contribution : f32,

    _padding1 : u32,
    _padding2 : u32,
};

struct Pixel {
    data: vec4<f32>,
    partitionNum: u32,

    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
};

struct InputBlock {
    pixels: array<Pixel, BLOCK_MAX_TEXELS>,
    partition_pixel_counts: array<u32, 4>,
    data_min: vec4<f32>,
    data_max: vec4<f32>,

    grayscale: u32,
    partitioning_idx: u32,
    xpos: u32,
    ypos: u32,
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



@group(0) @binding(0) var<uniform> uniforms: UniformVariables;
@group(0) @binding(1) var<storage, read> block_modes: array<BlockMode>;
@group(0) @binding(2) var<storage, read> decimation_infos: array<DecimationInfo>;
@group(0) @binding(3) var<storage, read> texel_to_weight_map: array<TexelToWeightMap>;
@group(0) @binding(4) var<storage, read> input_blocks: array<InputBlock>;
@group(0) @binding(5) var<storage, read> final_candidates: array<FinalCandidate>;
@group(0) @binding(6) var<storage, read> unpacked_endpoints: array<UnpackedEndpoints>;

@group(0) @binding(7) var<storage, read_write> output_final_errors: array<f32>;



var<workgroup> dec_weights: array<f32, BLOCK_MAX_WEIGHTS>;
var<workgroup> undec_weights: array<f32, BLOCK_MAX_TEXELS>;
var<workgroup> shared_total_error: atomic<u32>;


fn atomicAdd_f32(atomic_target: ptr<workgroup, atomic<u32>>, value_to_add: f32) {
    loop {
        let original_val_uint = atomicLoad(atomic_target);
        let original_val_float = bitcast<f32>(original_val_uint);
        let new_val_float = original_val_float + value_to_add;
        let new_val_uint = bitcast<u32>(new_val_float);
        let result = atomicCompareExchangeWeak(atomic_target, original_val_uint, new_val_uint);
        if (result.exchanged) {
            break;
        }
    }
}



@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(workgroup_id) group_id: vec3<u32>, @builtin(local_invocation_index) local_idx: u32) {
    let candidate_idx = group_id.x;

    let bm = block_modes[final_candidates[candidate_idx].block_mode_index];
    let di = decimation_infos[bm.decimation_mode];
    let block_idx = candidate_idx / uniforms.tune_candidate_limit;
    let partition_count = uniforms.partition_count;

    let input_block = input_blocks[block_idx];

    let quantized_weights = final_candidates[candidate_idx].quantized_weights;

    // Unquantize and bilinear infill
    for (var i = local_idx; i < di.weight_count; i += WORKGROUP_SIZE) {
        let weight_int = quantized_weights[i];
        dec_weights[i] = f32(weight_int) / 64.0;
    }
    workgroupBarrier();

    for (var i = local_idx; i < di.texel_count; i += WORKGROUP_SIZE) {
        if (di.texel_count == di.weight_count) {
            // Undecimated Case: Perform a direct copy.
            undec_weights[i] = dec_weights[i];
        } else {
            // Decimated Case: Perform the bilinear infill.
            var infill_val: f32 = 0.0;
            let weight_offset = di.texel_weights_offset[i];
            for (var j = 0u; j < di.texel_weight_count[i]; j = j + 1u) {
                let mapping = texel_to_weight_map[weight_offset + j];
                infill_val += dec_weights[mapping.weight_index] * mapping.contribution;
            }
            undec_weights[i] = infill_val;
        }
    }
    workgroupBarrier();

    //init sum variable
    if (local_idx == 0u) {
        atomicStore(&shared_total_error, bitcast<u32>(0.0f));
    }
    workgroupBarrier();

    //sum error for all texels
    for (var i = local_idx; i < di.texel_count; i += WORKGROUP_SIZE) {
        let p = input_block.pixels[i].partitionNum;
        if (p < partition_count) {

            let endpoint0 = unpacked_endpoints[candidate_idx].endpoint0[p];
            let endpoint1 = unpacked_endpoints[candidate_idx].endpoint1[p];

            let weight = undec_weights[i];

            let weight1 = vec4<i32>(i32(round(weight * 64.0)));
            let weight0 = vec4<i32>(64) - weight1;

            var color = (endpoint0 * weight0) + (endpoint1 * weight1) + vec4<i32>(32);
            color = color >> vec4<u32>(6);

            var diff = input_block.pixels[i].data - vec4<f32>(color);
            diff = min(abs(diff), vec4<f32>(1e15f));
            
            let error = dot(diff * diff, uniforms.channel_weights);

            atomicAdd_f32(&shared_total_error, min(error, ERROR_CALC_DEFAULT));
        }
    }
    workgroupBarrier();

    //store result
    if(local_idx == 0u) {
        output_final_errors[candidate_idx] = bitcast<f32>(atomicLoad(&shared_total_error));
    }

}