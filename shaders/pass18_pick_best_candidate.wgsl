const BLOCK_MAX_TEXELS: u32 = 144u;
const BLOCK_MAX_WEIGHTS: u32 = 64u;
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

struct SymbolicBlock {
    errorval: f32,

    block_mode_index: u32,
    partition_count: u32,
    partition_index: u32,

    partition_formats_matched: u32,
    quant_mode: u32,

    _padding1: u32,
    _padding2: u32,

    partition_formats: vec4<u32>,

    packed_color_values: array<u32, 32>, //8 integers per partition

    quantized_weights: array<u32, BLOCK_MAX_WEIGHTS>,
};



@group(0) @binding(0) var<uniform> uniforms: UniformVariables;
@group(0) @binding(1) var<storage, read> inputBlocks: array<InputBlock>;
@group(0) @binding(2) var<storage, read> top_candidates: array<FinalCandidate>;
@group(0) @binding(3) var<storage, read> block_modes: array<BlockMode>;

@group(0) @binding(4) var<storage, read_write> output_symbolic_blocks: array<SymbolicBlock>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    
    let block_idx = global_id.x;

    var best_error = ERROR_CALC_DEFAULT;
    var best_candidate_idx = 0u;

    for (var i = 0u; i < uniforms.tune_candidate_limit; i = i + 1u) {
        let candidate_idx = block_idx * uniforms.tune_candidate_limit + i;
        let current_error = top_candidates[candidate_idx].total_error;

        if (current_error < best_error) {
            best_error = current_error;
            best_candidate_idx = candidate_idx;
        }
    }

    let winner = top_candidates[best_candidate_idx];
    //let winner = top_candidates[block_idx + 0];
    let out_ptr = &output_symbolic_blocks[block_idx];

    (*out_ptr).errorval = best_error;
    (*out_ptr).block_mode_index = block_modes[winner.block_mode_index].mode_index;
    (*out_ptr).partition_count = uniforms.partition_count;
    (*out_ptr).partition_index = inputBlocks[block_idx].partitioning_idx;
    (*out_ptr).quant_mode = winner.final_quant_mode;
    (*out_ptr).partition_formats_matched = winner.color_formats_matched;

    (*out_ptr).partition_formats = winner.final_formats;
    (*out_ptr).packed_color_values = winner.packed_color_values;
    (*out_ptr).quantized_weights = winner.quantized_weights;
}
