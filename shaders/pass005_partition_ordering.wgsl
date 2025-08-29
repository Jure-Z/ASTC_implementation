const BLOCK_MAX_PARTITIONINGS: u32 = 1024u;
const SORT_ITEM_COUNT: u32 = BLOCK_MAX_PARTITIONINGS;
const WORKGROUP_SIZE: u32 = 256u;
const MAX_PARTITIONING_CANDIDATE_LIMIT: u32 = 512u;

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

    tune_partitoning_candidate_limit: u32,
    _padding1: u32,

    channel_weights : vec4<f32>,

    partitioning_count_selected : vec4<u32>,
    partitioning_count_all : vec4<u32>,
};

struct sortElement {
	score : u32,
	index : u32,
};


@group(0) @binding(0) var<uniform> uniforms: UniformVariables;
@group(0) @binding(1) var<storage, read> mismatch_counts : array<u32>;

@group(0) @binding(2) var<storage, read_write> partition_ordering : array<u32>;


var<workgroup> s_data: array<sortElement, SORT_ITEM_COUNT>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main( @builtin(workgroup_id) group_id: vec3<u32>, @builtin(local_invocation_index) local_idx: u32) {

    let block_idx = group_id.x;
    let partitioning_count_selected = uniforms.partitioning_count_selected[uniforms.partition_count - 1];

    // Load data into shared memory
    for(var i = local_idx; i < SORT_ITEM_COUNT; i += WORKGROUP_SIZE) {
        s_data[i].index = i;
        if(i < partitioning_count_selected) {
            let global_idx = block_idx * BLOCK_MAX_PARTITIONINGS + i;
            s_data[i].score = mismatch_counts[global_idx];
        }
        else {
			s_data[i].score = 0xFFFFFFFFu; // max value, so these will sort to the end
		}
    }
    workgroupBarrier();


    // Preform bitonic sort
    for (var k = 2u; k <= SORT_ITEM_COUNT; k = k * 2) {

        for (var j = k / 2u; j > 0u; j = j / 2u) {

			let partner_idx = local_idx ^ j;

			if (partner_idx > local_idx) {

				let ascending = ((local_idx & k) == 0u);

                let item1 = s_data[local_idx];
                let item2 = s_data[partner_idx];

                let should_swap = (item1.score > item2.score) || (item1.score == item2.score && item1.index > item2.index);

                if (should_swap == ascending) {
					s_data[local_idx] = item2;
					s_data[partner_idx] = item1;
				}
			}
			workgroupBarrier();
		}
    }

    //Write to output buffer
    for(var i = local_idx; i < uniforms.tune_partitoning_candidate_limit; i += WORKGROUP_SIZE) {
        let out_idx = block_idx * MAX_PARTITIONING_CANDIDATE_LIMIT + i;
        partition_ordering[out_idx] = s_data[i].index;
    }
}