const WORKGROUP_SIZE: u32 = 64u;
const BLOCK_MAX_TEXELS: u32 = 144u;
const BLOCK_MAX_WEIGHTS: u32 = 64u;

const TUNE_MAX_TRIAL_CANDIDATES = 8u;
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

struct PackedBlockModeLookup {
    block_mode_index: u32,
    decimation_mode_lookup_idx: u32, //index of corresponding decimation mode in the valid decimation modes buffer

    _padding1: u32,
    _padding2: u32,
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

struct QuantizationResult {
    error: f32,
    bitcount: i32,

    _padding1: u32,
    _padding2: u32,

    quantized_weights: array<u32, BLOCK_MAX_WEIGHTS>,
};

struct ColorCombinationResult {
    total_error: f32,
    best_quant_level: u32,
    best_quant_level_mod: u32,

    _padding1: u32,

    best_ep_formats: vec4<u32>,
};

struct SortItem {
	error: f32,
	bm_trial_idx: u32,
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

	final_formats: vec4<u32>,
	//8 integers per partition
    packed_color_values: array<u32, 32>,
};


@group(0) @binding(0) var<uniform> uniforms: UniformVariables;
@group(0) @binding(1) var<storage, read> valid_block_modes: array<PackedBlockModeLookup>;
@group(0) @binding(2) var<storage, read> ideal_endpoints_and_weights: array<IdealEndpointsAndWeights>;
@group(0) @binding(3) var<storage, read> quantization_results: array<QuantizationResult>;
@group(0) @binding(4) var<storage, read> color_combination_results: array<ColorCombinationResult>;

@group(0) @binding(5) var<storage, read_write> output_final_candidates: array<FinalCandidate>;
@group(0) @binding(6) var<storage, read_write> output_top_candidates: array<FinalCandidate>;


var<workgroup> topCandidates: array<SortItem, TUNE_MAX_TRIAL_CANDIDATES>;


@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) group_id: vec3<u32>, @builtin(local_invocation_index) local_idx: u32) {

    let block_idx = group_id.x;

    //initialize the top candidates array
    for(var i = 0u; i < TUNE_MAX_TRIAL_CANDIDATES; i += 1u) {
        topCandidates[i] = SortItem(ERROR_CALC_DEFAULT, 0u);
	}

    let start_idx = block_idx * uniforms.valid_block_mode_count;
    let trial_count = uniforms.valid_block_mode_count;
    let end_idx = start_idx + trial_count;

    for(var bm_trial_idx = start_idx; bm_trial_idx < end_idx; bm_trial_idx += 1) {
        
        let candidate_error = color_combination_results[bm_trial_idx].total_error;
        
        if (candidate_error < topCandidates[uniforms.tune_candidate_limit - 1u].error) {

            topCandidates[uniforms.tune_candidate_limit - 1u] = SortItem(candidate_error, bm_trial_idx);

            for (var j = uniforms.tune_candidate_limit - 1u; j > 0u; j = j - 1u) {
                if (topCandidates[j].error < topCandidates[j - 1u].error) {
                    let temp = topCandidates[j];
                    topCandidates[j] = topCandidates[j - 1u];
                    topCandidates[j - 1u] = temp;
                }
            }
        }
    }

    //Store the top N candidates
    for(var winner_idx = 0u; winner_idx < uniforms.tune_candidate_limit; winner_idx += 1u) {
        let winner = topCandidates[winner_idx];

        if (winner.error < ERROR_CALC_DEFAULT) {
            let bm_trial_idx = winner.bm_trial_idx;
            let winning_candidate = color_combination_results[bm_trial_idx];

            let bm_lookup_idx = bm_trial_idx % uniforms.valid_block_mode_count;

            let output_idx = block_idx * uniforms.tune_candidate_limit + winner_idx;
            let out_ptr = &output_final_candidates[output_idx];

			(*out_ptr).block_mode_index = valid_block_modes[bm_lookup_idx].block_mode_index;
			(*out_ptr).block_mode_trial_index = bm_trial_idx;
			(*out_ptr).total_error = winning_candidate.total_error;
			(*out_ptr).quant_level = winning_candidate.best_quant_level;
			(*out_ptr).quant_level_mod = winning_candidate.best_quant_level_mod;
			(*out_ptr).formats = winning_candidate.best_ep_formats;

            (*out_ptr).quantized_weights = quantization_results[bm_trial_idx].quantized_weights;
            (*out_ptr).candidate_partitions = ideal_endpoints_and_weights[block_idx].partitions;


        }

        //initialize top candidate errors to max error
        let output_idx = block_idx * uniforms.tune_candidate_limit + winner_idx;
        output_top_candidates[output_idx].total_error = ERROR_CALC_DEFAULT;
	}

}