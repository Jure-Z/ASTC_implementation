const BLOCK_MAX_TEXELS: u32 = 144u;
const MAX_PARTITIONS: u32 = 4u;
const WORKGROUP_SIZE: u32 = 256u;
const BLOCK_MAX_PARTITIONINGS: u32 = 1024u;
const MAX_PARTITIONING_CANDIDATE_LIMIT: u32 = 512u;

const FIXED_POINT_SCALE_I32: f32 = 4096.0;


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


@group(0) @binding(0) var<uniform> uniforms: UniformVariables;
@group(0) @binding(1) var<storage, read> partitionInfos: array<PartitonInfo>;
@group(0) @binding(2) var<storage, read> inputBlocks: array<InputBlock>;
@group(0) @binding(3) var<storage, read> partition_ordering : array<u32>;

@group(0) @binding(4) var<storage, read_write> final_partitioning_errors: array<vec2<f32>>;



var<workgroup> pixels: array<vec4<f32>, BLOCK_MAX_TEXELS>;


var<workgroup> partition_sums: array<array<atomic<u32>, 4>, MAX_PARTITIONS>;
var<workgroup> partition_counts: array<atomic<u32>, MAX_PARTITIONS>;

// For direction computation: [partition_idx][axis_sum_idx][color_channel]
// axis_sum_idx: 0=sum_xp, 1=sum_yp, 2=sum_zp, 3=sum_wp
var<workgroup> axis_sums: array<array<array<atomic<i32>, 4>, 4>, MAX_PARTITIONS>;

var<workgroup> uncor_error_sum: atomic<u32>;
var<workgroup> samec_error_sum: atomic<u32>;

var<workgroup> line_min_param: array<atomic<i32>, MAX_PARTITIONS>;
var<workgroup> line_max_param: array<atomic<i32>, MAX_PARTITIONS>;


@compute @workgroup_size(WORKGROUP_SIZE)
fn main( @builtin(workgroup_id) group_id: vec3<u32>, @builtin(local_invocation_index) local_idx: u32) {
    
    let block_idx = group_id.x;

    //load pixel data into shared memory
    if (local_idx < uniforms.texel_count) {
		pixels[local_idx] = inputBlocks[block_idx].pixels[local_idx];
	}
    workgroupBarrier();

    //iterate over all candidate partitionings to test
    for(var cand_idx = 0u; cand_idx < uniforms.tune_partitoning_candidate_limit; cand_idx += 1u) {
        
        let global_idx = block_idx * MAX_PARTITIONING_CANDIDATE_LIMIT + cand_idx;
        let partitioning_idx = partition_ordering[global_idx];
        let table_idx = (uniforms.partition_count - 2) * BLOCK_MAX_PARTITIONINGS + partitioning_idx;
        let pi = partitionInfos[table_idx];


        //1. Compute averages
        //initialize atomics
        if(local_idx < uniforms.partition_count) {
            atomicStore(&partition_counts[local_idx], 0u);
            for(var c = 0u; c < 4u; c += 1u) {
				atomicStore(&partition_sums[local_idx][c], 0u);
			}
        }
        workgroupBarrier();

        //accumulate sums and counts
        if(local_idx < uniforms.texel_count) {
			let pidx = pi.partition_of_texel[local_idx];
			let px = pixels[local_idx];
            let px_u = vec4<u32>(px);

			atomicAdd(&partition_counts[pidx], 1u);
            atomicAdd(&partition_sums[pidx][0], px_u.r);
            atomicAdd(&partition_sums[pidx][1], px_u.g);
            atomicAdd(&partition_sums[pidx][2], px_u.b);
            atomicAdd(&partition_sums[pidx][3], px_u.a);
		}
        workgroupBarrier();

        //finalize averages
        var avg: array<vec4<f32>, MAX_PARTITIONS>;
        if(local_idx < uniforms.partition_count) {
            let count = f32(atomicLoad(&partition_counts[local_idx]));
            if(count > 0.0) {
                avg[local_idx] = vec4<f32>(
					f32(atomicLoad(&partition_sums[local_idx][0])) / count,
					f32(atomicLoad(&partition_sums[local_idx][1])) / count,
					f32(atomicLoad(&partition_sums[local_idx][2])) / count,
					f32(atomicLoad(&partition_sums[local_idx][3])) / count
				);            
            }
        }
        workgroupBarrier();

        //2. Compute directions
        //initialize atomics
        if(local_idx < uniforms.partition_count) {
            for (var axis = 0u; axis < 4u; axis += 1u) {
                for (var c = 0u; c < 4u; c += 1u) {
                    atomicStore(&axis_sums[local_idx][axis][c], 0);
                }
            }
        }
        workgroupBarrier();

        //accumulate axis sums
        if(local_idx < uniforms.texel_count) {
            let pidx = pi.partition_of_texel[local_idx];

            let datum = pixels[local_idx] - avg[pidx];
            let datum_i = vec4<i32>(datum);

            if(datum.x > 0.0) {
                atomicAdd(&axis_sums[pidx][0][0], datum_i.r);
                atomicAdd(&axis_sums[pidx][0][1], datum_i.g);
                atomicAdd(&axis_sums[pidx][0][2], datum_i.b);
                atomicAdd(&axis_sums[pidx][0][3], datum_i.a);
            }
            if(datum.y > 0.0) {
                atomicAdd(&axis_sums[pidx][1][0], datum_i.r);
                atomicAdd(&axis_sums[pidx][1][1], datum_i.g);
                atomicAdd(&axis_sums[pidx][1][2], datum_i.b);
                atomicAdd(&axis_sums[pidx][1][3], datum_i.a);
            }
            if(datum.z > 0.0) {
                atomicAdd(&axis_sums[pidx][2][0], datum_i.r);
                atomicAdd(&axis_sums[pidx][2][1], datum_i.g);
                atomicAdd(&axis_sums[pidx][2][2], datum_i.b);
                atomicAdd(&axis_sums[pidx][2][3], datum_i.a);
            }
            if(datum.w > 0.0) {
                atomicAdd(&axis_sums[pidx][3][0], datum_i.r);
                atomicAdd(&axis_sums[pidx][3][1], datum_i.g);
                atomicAdd(&axis_sums[pidx][3][2], datum_i.b);
                atomicAdd(&axis_sums[pidx][3][3], datum_i.a);
            }
        }
        workgroupBarrier();

        //finalize directions
        var dir: array<vec4<f32>, MAX_PARTITIONS>;
        if(local_idx < uniforms.partition_count) {
            let p = local_idx;
            let sum_xp = vec4<f32>(vec4<i32>(atomicLoad(&axis_sums[p][0][0]), atomicLoad(&axis_sums[p][0][1]), atomicLoad(&axis_sums[p][0][2]), atomicLoad(&axis_sums[p][0][3])));
            let sum_yp = vec4<f32>(vec4<i32>(atomicLoad(&axis_sums[p][1][0]), atomicLoad(&axis_sums[p][1][1]), atomicLoad(&axis_sums[p][1][2]), atomicLoad(&axis_sums[p][1][3])));
            let sum_zp = vec4<f32>(vec4<i32>(atomicLoad(&axis_sums[p][2][0]), atomicLoad(&axis_sums[p][2][1]), atomicLoad(&axis_sums[p][2][2]), atomicLoad(&axis_sums[p][2][3])));
            let sum_wp = vec4<f32>(vec4<i32>(atomicLoad(&axis_sums[p][3][0]), atomicLoad(&axis_sums[p][3][1]), atomicLoad(&axis_sums[p][3][2]), atomicLoad(&axis_sums[p][3][3])));

            var best_sum = sum_xp;
            var max_prod = dot(sum_xp, sum_xp);
            
            var prod = dot(sum_yp, sum_yp);
            if(prod > max_prod) {
				max_prod = prod;
				best_sum = sum_yp;
			}
            prod = dot(sum_zp, sum_zp);
			if(prod > max_prod) {
                max_prod = prod;
                best_sum = sum_zp;
            }
            prod = dot(sum_wp, sum_wp);
            if(prod > max_prod) {
				max_prod = prod;
				best_sum = sum_wp;
			}

            dir[p] = normalize(best_sum);
        }
        workgroupBarrier();


        //3. Compute errors
        //initialize atomics
        if(local_idx == 0u) {
            atomicStore(&uncor_error_sum, 0u);
            atomicStore(&samec_error_sum, 0u);
        }
        if(local_idx < uniforms.partition_count) {
            atomicStore(&line_min_param[local_idx], 2147483647);
            atomicStore(&line_max_param[local_idx], -2147483647);
		}
        workgroupBarrier();


        //Each thread calculates it's texels contrubution to the error
        if(local_idx < uniforms.texel_count) {
            let pidx = pi.partition_of_texel[local_idx];
            let px = pixels[local_idx];

            //uncorrelated line
            let uncor_dir = dir[pidx];
            let uncor_param = dot(px, uncor_dir);
            let uncor_proj = avg[pidx] + uncor_dir * dot(px - avg[pidx], uncor_dir);
            let uncor_diff = px - uncor_proj;
            let uncor_dist_sq = dot(uncor_diff * uncor_diff, uniforms.channel_weights);

            //same chroma line
            let samec_dir = normalize(avg[pidx]);
            let samec_proj = samec_dir * dot(px, samec_dir);
            let samec_diff = px - samec_proj;
            let samec_dist_sq = dot(samec_diff * samec_diff, uniforms.channel_weights);

            atomicAdd(&uncor_error_sum, u32(uncor_dist_sq));
            atomicAdd(&samec_error_sum, u32(samec_dist_sq));

            //update line params
            let uncor_param_i = i32(uncor_param * FIXED_POINT_SCALE_I32);
            atomicMin(&line_min_param[pidx], uncor_param_i);
            atomicMax(&line_max_param[pidx], uncor_param_i);
        }
        workgroupBarrier();

        //finalize error
        if(local_idx == 0u) {
            var total_uncor_err = f32(atomicLoad(&uncor_error_sum));
            var total_samec_err = f32(atomicLoad(&samec_error_sum));

            for(var p = 0u; p < uniforms.partition_count; p += 1u) {
                let min_p = f32(atomicLoad(&line_min_param[p])) / FIXED_POINT_SCALE_I32;
                let max_p = f32(atomicLoad(&line_max_param[p])) / FIXED_POINT_SCALE_I32;
                let line_len = max(max_p - min_p, 1e-7);

                let tpp = f32(pi.partition_texel_count[p]);

                let texels_per_block = uniforms.texel_count;
                var weight_imprecision_estim = 0.055f;
                if(texels_per_block <= 20) {
                    weight_imprecision_estim = 0.03f;
                }
                else if(texels_per_block <= 31) {
                    weight_imprecision_estim = 0.04f;
                }
                else if(texels_per_block <= 41) {
                    weight_imprecision_estim = 0.05f;
                }
                weight_imprecision_estim = weight_imprecision_estim * weight_imprecision_estim;

                let error_weight = tpp * weight_imprecision_estim;

                total_uncor_err = total_uncor_err + error_weight * (line_len * line_len);
                total_samec_err = total_samec_err + error_weight * dot(avg[p], avg[p]);
            }

            let out_idx = block_idx * MAX_PARTITIONING_CANDIDATE_LIMIT + cand_idx;
            final_partitioning_errors[out_idx] = vec2<f32>(total_uncor_err, total_samec_err);
        }
        workgroupBarrier();
    }
}