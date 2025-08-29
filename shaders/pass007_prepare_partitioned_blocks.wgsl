const BLOCK_MAX_TEXELS: u32 = 144u;
const MAX_PARTITIONS: u32 = 4u;

const WORKGROUP_SIZE: u32 = 64u;

const BLOCK_MAX_PARTITIONINGS: u32 = 1024u;
const MAX_PARTITIONING_CANDIDATE_LIMIT: u32 = 512u;
const MAX_PARTITIONINGS: u32 = 8u;

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
    requested_partitionings: u32,

    channel_weights : vec4<f32>,

    partitioning_count_selected : vec4<u32>,
    partitioning_count_all : vec4<u32>,
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

struct PartitonInfo {
    partition_count: u32,
    partition_index: u32,
    _padding1: u32,
    _padding2: u32,

    partition_texel_count: array<u32, MAX_PARTITIONS>,
    partition_of_texel: array<u32, BLOCK_MAX_TEXELS>,
};


struct BestChoice {
    index: u32,
    error: f32,
};


@group(0) @binding(0) var<uniform> uniforms: UniformVariables;
@group(0) @binding(1) var<storage, read> partitionInfos: array<PartitonInfo>;
@group(0) @binding(2) var<storage, read> inputBlocks: array<InputBlock>;
@group(0) @binding(3) var<storage, read> final_partitioning_errors: array<vec2<f32>>;

@group(0) @binding(4) var<storage, read_write> partitionedBlocks: array<InputBlock>;


@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {

    let blockIndex = global_id.x;


    var best_uncor: array<BestChoice, MAX_PARTITIONINGS>;
    var best_samec: array<BestChoice, MAX_PARTITIONINGS>;

    // Initialize best choices
    for (var i = 0u; i < MAX_PARTITIONINGS; i += 1u) {
        best_uncor[i] = BestChoice(0u, 1e30);
        best_samec[i] = BestChoice(0u, 1e30);
    }

    //find N best partitionings for uncorelated and same chroma colors
    for(var i = 0u; i < uniforms.tune_partitoning_candidate_limit; i += 1u) {
        let global_idx = blockIndex * MAX_PARTITIONING_CANDIDATE_LIMIT + i;
        let errors = final_partitioning_errors[global_idx];

        //uncorrelated
        if(errors.x < best_uncor[uniforms.requested_partitionings - 1u].error) {
            best_uncor[uniforms.requested_partitionings - 1u] = BestChoice(i, errors.x);

            //simple bubble-up sort
            for(var j = uniforms.requested_partitionings - 1u; j > 0u; j -= 1u) {
				if(best_uncor[j].error < best_uncor[j - 1u].error) {
					let tmp = best_uncor[j - 1u];
					best_uncor[j - 1u] = best_uncor[j];
					best_uncor[j] = tmp;
				}
			}
        }

        //same chroma
        if(errors.y < best_samec[uniforms.requested_partitionings - 1u].error) {
			best_samec[uniforms.requested_partitionings - 1u] = BestChoice(i, errors.y);

			//simple bubble-up sort
			for(var j = uniforms.requested_partitionings - 1u; j > 0u; j -= 1u) {
                if(best_samec[j].error < best_samec[j - 1u].error) {
					let tmp = best_samec[j - 1u];
					best_samec[j - 1u] = best_samec[j];
					best_samec[j] = tmp;
				}
            }
        }
    }


    //Interleave and deduplicate
    var interleave: array<u32, MAX_PARTITIONINGS * 2u>;
    for(var i = 0u; i < uniforms.requested_partitionings; i += 1u) {
		interleave[i * 2u] = best_uncor[i].index;
		interleave[i * 2u + 1u] = best_samec[i].index;
	}

    var final_indices: array<u32, MAX_PARTITIONINGS>;
    var emitted = 0u;

    for(var i = 0u; i < uniforms.requested_partitionings * 2u; i += 1u) {
        let idx = interleave[i];

        var written = false;
        for(var j = 0u; j < emitted; j += 1u) {
			if(final_indices[j] == idx) {
				written = true;
				break;
			}
		}

        if(!written) {
            final_indices[emitted] = idx;
            emitted += 1u;
            if(emitted >= uniforms.requested_partitionings) {
				break;
			}
        }
    }
    let num_final = emitted;

    //generate partitoned blocks
    let original_block = inputBlocks[blockIndex];

    for(var i = 0u; i < num_final; i += 1u) {
        let part_idx = final_indices[i];
        let table_idx = (uniforms.partition_count - 2) * BLOCK_MAX_PARTITIONINGS + part_idx;
        let pi = partitionInfos[table_idx];

        var new_block: InputBlock;

        new_block.pixels = original_block.pixels;
        new_block.grayscale = original_block.grayscale;
        new_block.constant_alpha = original_block.constant_alpha;

        new_block.partitioning_idx = pi.partition_index;
        new_block.partition_pixel_counts = pi.partition_texel_count;
        new_block.texel_partitions = pi.partition_of_texel;

        let out_idx = blockIndex * uniforms.requested_partitionings + i;
        partitionedBlocks[out_idx] = new_block;

    }

}