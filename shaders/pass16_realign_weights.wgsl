const WORKGROUP_SIZE: u32 = 64u;
const BLOCK_MAX_TEXELS: u32 = 144u;
const BLOCK_MAX_WEIGHTS: u32 = 64u;
const BLOCK_MAX_PARTITIONS: u32 = 4u;

const TUNE_MAX_TRIAL_CANDIDATES = 8u;

//A table of previous and next weights, indexed by current value
//bits 7:0 previous value
//bits 15:8 next value
//table is flattened. Index is: quant_level * 65 + value
//table size is: 12 (QUANT_2 up to QUANT_32) * 65 (values from 0 up to 64) = 780
const ALL_PREV_NEXT_VALUES = array<u32, 780>(
    //QUANT_2
    0x4000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0x4000,
    //QUANT_3
    0x2000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0x4000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0x4020,
    //QUANT_4
    0x1500,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0x2b00,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0x4015,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0x402b,
    //QUANT_5
    0x1000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0x2000,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0x3010,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0x4020,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0x4030,
    //QUANT_6
    0x0c00,0,0,0,0,0,0,0,0,0,0,0,0x1900,0,0,0,0,0,0,0,0,0,0,0,0,
    0x270c,0,0,0,0,0,0,0,0,0,0,0,0,0,0x3419,0,0,0,0,0,0,0,0,0,0,
    0,0,0x4027,0,0,0,0,0,0,0,0,0,0,0,0x4034,
    //QUANT_8
    0x0900,0,0,0,0,0,0,0,0,0x1200,0,0,0,0,0,0,0,0,0x1b09,0,0,
    0,0,0,0,0,0,0x2512,0,0,0,0,0,0,0,0,0,0x2e1b,0,0,0,0,0,0,0,0,
    0x3725,0,0,0,0,0,0,0,0,0x402e,0,0,0,0,0,0,0,0,0x4037,
    //QUANT_10
    0x0700,0,0,0,0,0,0,0x0e00,0,0,0,0,0,0,0x1507,0,0,0,0,0,0,
    0x1c0e,0,0,0,0,0,0,0x2415,0,0,0,0,0,0,0,0x2b1c,0,0,0,0,0,
    0,0x3224,0,0,0,0,0,0,0x392b,0,0,0,0,0,0,0x4032,0,0,0,0,0,
    0,0x4039,
    //QUANT_12
    0x0500,0,0,0,0,0x0b00,0,0,0,0,0,0x1105,0,0,0,0,0,
    0x170b,0,0,0,0,0,0x1c11,0,0,0,0,0x2417,0,0,0,0,0,0,0,
    0x291c,0,0,0,0,0x2f24,0,0,0,0,0,0x3529,0,0,0,0,0,
    0x3b2f,0,0,0,0,0,0x4035,0,0,0,0,0x403b,
    //QUANT_16
    0x0400,0,0,0,0x0800,0,0,0,0x0c04,0,0,0,0x1108,0,0,0,0,
    0x150c,0,0,0,0x1911,0,0,0,0x1d15,0,0,0,0x2319,0,0,0,0,
    0,0x271d,0,0,0,0x2b23,0,0,0,0x2f27,0,0,0,0x342b,0,0,0,
    0,0x382f,0,0,0,0x3c34,0,0,0,0x4038,0,0,0,0x403c,
    //QUANT_20
    0x0300,0,0,0x0600,0,0,0x0903,0,0,0x0d06,0,0,0,
    0x1009,0,0,0x130d,0,0,0x1710,0,0,0,0x1a13,0,0,
    0x1d17,0,0,0x231a,0,0,0,0,0,0x261d,0,0,0x2923,0,0,
    0x2d26,0,0,0,0x3029,0,0,0x332d,0,0,0x3730,0,0,0,
    0x3a33,0,0,0x3d37,0,0,0x403a,0,0,0x403d,
    //QUANT_24
    0x0200,0,0x0500,0,0,0x0802,0,0,0x0b05,0,0,0x0d08,
    0,0x100b,0,0,0x130d,0,0,0x1610,0,0,0x1813,0,
    0x1b16,0,0,0x1e18,0,0,0x221b,0,0,0,0x251e,0,0,
    0x2822,0,0,0x2a25,0,0x2d28,0,0,0x302a,0,0,0x332d,
    0,0,0x3530,0,0x3833,0,0,0x3b35,0,0,0x3e38,0,0,
    0x403b,0,0x403e,
    //QUNAT_32
    0x0200,0,0x0400,0,0x0602,0,0x0804,0,0x0a06,0,
    0x0c08,0,0x0e0a,0,0x100c,0,0x120e,0,0x1410,0,
    0x1612,0,0x1814,0,0x1a16,0,0x1c18,0,0x1e1a,0,
    0x221c,0,0,0,0x241e,0,0x2622,0,0x2824,0,0x2a26,0,
    0x2c28,0,0x2e2a,0,0x302c,0,0x322e,0,0x3430,0,
    0x3632,0,0x3834,0,0x3a36,0,0x3c38,0,0x3e3a,0,
    0x403c,0,0x403e
);

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

struct WeightToTexelMap {
    texel_index : u32,
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

struct UnpackedEndpoints {
    endpoint0: array<vec4<i32>, 4>,
    endpoint1: array<vec4<i32>, 4>,
}

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



@group(0) @binding(0) var<uniform> uniforms: UniformVariables;
@group(0) @binding(1) var<storage, read> block_modes: array<BlockMode>;
@group(0) @binding(2) var<storage, read> decimation_infos: array<DecimationInfo>;
@group(0) @binding(3) var<storage, read> texel_to_weight_map: array<TexelToWeightMap>;
@group(0) @binding(4) var<storage, read> weight_to_texel_map: array<WeightToTexelMap>;
@group(0) @binding(5) var<storage, read> input_blocks: array<InputBlock>;
@group(0) @binding(6) var<storage, read> unpacked_endpoints: array<UnpackedEndpoints>;

@group(0) @binding(7) var<storage, read_write> final_candidates: array<FinalCandidate>;



var<workgroup> uq_weightsf: array<f32, BLOCK_MAX_WEIGHTS>; 

var<workgroup> part_offsets: array<vec4<f32>, 4>;
var<workgroup> part_bases: array<vec4<f32>, 4>;



@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(workgroup_id) group_id: vec3<u32>, @builtin(local_invocation_index) local_idx: u32) {

    let block_idx = group_id.x;
    let candidate_idx = block_idx * uniforms.tune_candidate_limit + group_id.y;

    let bm = block_modes[final_candidates[candidate_idx].block_mode_index];
    let di = decimation_infos[bm.decimation_mode];
    let input_block = input_blocks[block_idx];
    let partition_count = uniforms.partition_count;


    //Unpack endpoints and pre-calculate offset vectors for all partitions.
    if (local_idx < partition_count) {
        let p = local_idx;
        let ep0 = unpacked_endpoints[candidate_idx].endpoint0[p];
        let ep1 = unpacked_endpoints[candidate_idx].endpoint1[p];

        let delta = ep1 - ep0;

        part_bases[p] = vec4<f32>(ep0);
        // The offset is the endpoint delta scaled by 1/64 for unquantization
        part_offsets[p] = vec4<f32>(delta) * (1.0 / 64.0);
    }
    workgroupBarrier();

    //Unquantize all weights to a shared float array (used for decimated weights)
    for (var i = local_idx; i < di.weight_count; i += WORKGROUP_SIZE) {
        uq_weightsf[i] = f32(final_candidates[candidate_idx].quantized_weights[i]);
    }

    workgroupBarrier();


    //Branch between decimated and undecimated logic
    if(di.weight_count == di.texel_count) {
        //realign weights undecimated

        for (var texel_idx = local_idx; texel_idx < di.texel_count; texel_idx += WORKGROUP_SIZE) {

            let candidate_weight_ptr = &final_candidates[candidate_idx].quantized_weights[texel_idx];
            let uqw = i32(*candidate_weight_ptr);

            // Look up the previous and next quantization steps
            let prev_next = ALL_PREV_NEXT_VALUES[bm.quant_mode * 65u + u32(uqw)];
            let uqw_down = i32(prev_next & 0xFFu);
            let uqw_up = i32((prev_next >> 8u) & 0xFFu);

            let weight_base = f32(uqw);
            let weight_down_diff = f32(uqw_down - uqw);
            let weight_up_diff = f32(uqw_up - uqw);

            let p = input_block.pixels[texel_idx].partitionNum;
            let color_offset = part_offsets[p];
            let color_base = part_bases[p];
            
            let color = color_base + color_offset * weight_base;
            let orig_color = input_block.pixels[texel_idx].data;
            let error_weight = uniforms.channel_weights;

            let color_diff = color - orig_color;
            let color_diff_down = color_diff + color_offset * weight_down_diff;
            let color_diff_up = color_diff + color_offset * weight_up_diff;
            
            let error_base = dot(color_diff * color_diff, error_weight);
            let error_down = dot(color_diff_down * color_diff_down, error_weight);
            let error_up = dot(color_diff_up * color_diff_up, error_weight);

            // Check if moving the weight up or down improves the error
            if ((error_up < error_base) && (error_up < error_down) && (uqw < 64)) {
                *candidate_weight_ptr = u32(uqw_up);
            } else if ((error_down < error_base) && (uqw > 0)) {
                *candidate_weight_ptr = u32(uqw_down);
            }
        }
    }
    else {
        //realign weights decimated

        if(local_idx == 0) {
        for (var we_idx = 0u; we_idx < di.weight_count; we_idx += 1u) {

            let candidate_weight_ptr = &final_candidates[candidate_idx].quantized_weights[we_idx];
            let uqw = i32(*candidate_weight_ptr);
            
            let prev_next = ALL_PREV_NEXT_VALUES[bm.quant_mode * 65u + u32(uqw)];
            let uqw_down = f32(prev_next & 0xFFu);
            let uqw_up = f32((prev_next >> 8u) & 0xFFu);
            let uqw_base = f32(uqw);

            let uqw_diff_down = uqw_down - uqw_base;
            let uqw_diff_up = uqw_up - uqw_base;
            
            var error_basev = vec4<f32>(0.0);
            var error_downv = vec4<f32>(0.0);
            var error_upv = vec4<f32>(0.0);

            // Interpolate colors to get the diffs
            let texels_to_eval = di.weight_texel_count[we_idx];
            for (var te_idx = 0u; te_idx < texels_to_eval; te_idx = te_idx + 1u) {
                let wt_map = weight_to_texel_map[di.weight_texels_offset[we_idx] + te_idx];
                let texel_idx = wt_map.texel_index;

                // Perform bilinear infill for this texel
                var weight_base = 0.0;
                let tw_offset = di.texel_weights_offset[texel_idx];
                for (var j = 0u; j < di.texel_weight_count[texel_idx]; j = j + 1u) {
                    let tw_map = texel_to_weight_map[tw_offset + j];
                    weight_base += uq_weightsf[tw_map.weight_index] * tw_map.contribution;
                }

                var tw_base = 0.0;
                let tw_count = di.texel_weight_count[texel_idx];
                for (var k = 0u; k < tw_count; k = k + 1u) {
                    let tw_map = texel_to_weight_map[tw_offset + k];
                    if (tw_map.weight_index == we_idx) {
                        tw_base = tw_map.contribution;
                        break;
                    }
                }

                let weight_down_diff = uqw_diff_down * tw_base;
                let weight_up_diff = uqw_diff_up * tw_base;

                let p = input_block.pixels[texel_idx].partitionNum;
                let color_offset = part_offsets[p];
                let color_base = part_bases[p];

                let color = color_base + color_offset * weight_base;
                let orig_color = input_block.pixels[texel_idx].data;

                let color_diff = color - orig_color;
                let color_diff_down = color_diff + color_offset * weight_down_diff;
                let color_diff_up = color_diff + color_offset * weight_up_diff;
                
                error_basev += color_diff * color_diff;
                error_downv += color_diff_down * color_diff_down;
                error_upv += color_diff_up * color_diff_up;
            }

            let error_weight = uniforms.channel_weights;
            let error_base = dot(error_basev, error_weight);
            let error_down = dot(error_downv, error_weight);
            let error_up = dot(error_upv, error_weight);

            if ((error_up < error_base) && (error_up < error_down) && (uqw < 64)) {
                *candidate_weight_ptr = u32(uqw_up);
                uq_weightsf[we_idx] = uqw_up;
            } else if ((error_down < error_base) && (uqw > 0)) {
                *candidate_weight_ptr = u32(uqw_down);
                uq_weightsf[we_idx] = uqw_down;
            }
        }
        }
    }

}
