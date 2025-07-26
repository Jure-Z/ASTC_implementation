const WORKGROUP_SIZE: u32 = 64u;

const TUNE_MAX_TRIAL_CANDIDATES = 8u;
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

struct BlockModeTrial {
    block_index : u32,
    block_mode_index : u32,
    decimation_mode_trial_index : u32,

    _padding1 : u32,
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
	original_trial_idx: u32,
};

struct FinalCandidate {
    block_mode_index: u32,
    block_mode_trial_index: u32,
    total_error: f32,
    quant_level: u32,
    quant_level_mod: u32,

    _padding1: u32,
    _padding2: u32,
    _padding3: u32,

    formats: vec4<u32>,
};


@group(0) @binding(0) var<uniform> uniforms: UniformVariables;
@group(0) @binding(1) var<storage, read> block_mode_trials: array<BlockModeTrial>;
@group(0) @binding(2) var<storage, read> modes_per_block: array<u32>;
@group(0) @binding(3) var<storage, read> block_mode_trial_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> color_combination_results: array<ColorCombinationResult>;

@group(0) @binding(5) var<storage, read_write> output_final_candidates: array<FinalCandidate>;


var<workgroup> topCandidates: array<SortItem, TUNE_MAX_TRIAL_CANDIDATES>;


@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) group_id: vec3<u32>, @builtin(local_invocation_index) local_idx: u32) {

    let block_idx = group_id.x;

    //initialize the top candidates array
    for(var i = 0u; i < TUNE_MAX_TRIAL_CANDIDATES; i += 1u) {
        topCandidates[i] = SortItem(ERROR_CALC_DEFAULT, 0u);
	}

    let start_idx = block_mode_trial_offsets[block_idx];
    let trial_count = modes_per_block[block_idx];
    let end_idx = start_idx + trial_count;

    for(var trial_idx = start_idx; trial_idx < end_idx; trial_idx += 1) {
        
        let candidate_error = color_combination_results[trial_idx].total_error;
        
        if (candidate_error < topCandidates[uniforms.tune_candidate_limit - 1u].error) {

            topCandidates[uniforms.tune_candidate_limit - 1u] = SortItem(candidate_error, trial_idx);

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
            let trial_idx = winner.original_trial_idx;
            let winning_trial = block_mode_trials[trial_idx];
            let winning_candidate = color_combination_results[trial_idx];

            let output_idx = block_idx * uniforms.tune_candidate_limit + winner_idx;
            let out_ptr = &output_final_candidates[output_idx];

			(*out_ptr).block_mode_index = winning_trial.block_mode_index;
			(*out_ptr).block_mode_trial_index = trial_idx;
			(*out_ptr).total_error = winning_candidate.total_error;
			(*out_ptr).quant_level = winning_candidate.best_quant_level;
			(*out_ptr).quant_level_mod = winning_candidate.best_quant_level_mod;
			(*out_ptr).formats = winning_candidate.best_ep_formats;
        }
	}
}