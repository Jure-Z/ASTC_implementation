const BLOCK_MAX_TEXELS: u32 = 144u;
const KMEANS_TEXELS: u32 = 64u;
const WORKGROUP_SIZE: u32 = 256u;
const BLOCK_MAX_PARTITIONINGS: u32 = 1024u;


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

    partitioning_count_selected : vec4<u32>,
    partitioning_count_all : vec4<u32>,
};



@group(0) @binding(0) var<uniform> uniforms: UniformVariables;
@group(0) @binding(1) var<storage, read> kmeans_texels : array<u32, KMEANS_TEXELS>;
@group(0) @binding(2) var<storage, read> coverage_bitmaps_2 : array<vec2<u32>, 2 * BLOCK_MAX_PARTITIONINGS>;
@group(0) @binding(3) var<storage, read> coverage_bitmaps_3 : array<vec2<u32>, 3 * BLOCK_MAX_PARTITIONINGS>;
@group(0) @binding(4) var<storage, read> coverage_bitmaps_4 : array<vec2<u32>, 4 * BLOCK_MAX_PARTITIONINGS>;
@group(0) @binding(5) var<storage, read> texel_assignments : array<u32>;

@group(0) @binding(6) var<storage, read_write> mismatch_counts : array<u32>;


//split into two arrays, since atomic doesn't work vith vec2
var<workgroup> kmeans_bitmasks_high: array<atomic<u32>, 4>;
var<workgroup> kmeans_bitmasks_low: array<atomic<u32>, 4>;


fn popcount(v_in: u32) -> u32 {
    let mask1 = 0x55555555u;
    let mask2 = 0x33333333u;
    let mask3 = 0x0F0F0F0Fu;
    var v = v_in;

    v -= (v >> 1u) & mask1;
    v = (v & mask2) + ((v >> 2u) & mask2);
    v = (v + (v >> 4u)) & mask3;
    v *= 0x01010101u;
    v = v >> 24u;
    return v;
}

fn popcount64(v: vec2<u32>) -> u32 {
    return popcount(v.x) + popcount(v.y);
}

fn xor64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
	return vec2<u32>(a.x ^ b.x, a.y ^ b.y);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main( @builtin(workgroup_id) group_id: vec3<u32>, @builtin(local_invocation_index) local_idx: u32) {
    
    let block_idx = group_id.x;

    //build k-means bitmasks
    if (local_idx < uniforms.partition_count) {
        atomicStore(&kmeans_bitmasks_high[local_idx], 0u);
        atomicStore(&kmeans_bitmasks_low[local_idx], 0u);
    }
    workgroupBarrier();
    if(local_idx < KMEANS_TEXELS) {
        let texel_idx = kmeans_texels[local_idx];
        let global_idx = block_idx * BLOCK_MAX_TEXELS + texel_idx;
        let partition_assignment = texel_assignments[global_idx];
        let bit_to_set = 1u << (local_idx % 32u);
        if(local_idx < 32u) {
            atomicOr(&kmeans_bitmasks_low[partition_assignment], bit_to_set);
		} else {
			atomicOr(&kmeans_bitmasks_high[partition_assignment], bit_to_set);
		}
    }
    workgroupBarrier();


    let partition_count = uniforms.partition_count;
    let partitioning_count_selected = uniforms.partitioning_count_selected[partition_count - 1u];

    for(var i = local_idx; i < partitioning_count_selected; i += WORKGROUP_SIZE) {
        
        var mismatch_count = 0u;
        
        let a0 = vec2<u32>(atomicLoad(&kmeans_bitmasks_low[0]), atomicLoad(&kmeans_bitmasks_high[0]));
        let a1 = vec2<u32>(atomicLoad(&kmeans_bitmasks_low[1]), atomicLoad(&kmeans_bitmasks_high[1]));
        let a2 = vec2<u32>(atomicLoad(&kmeans_bitmasks_low[2]), atomicLoad(&kmeans_bitmasks_high[2]));
        let a3 = vec2<u32>(atomicLoad(&kmeans_bitmasks_low[3]), atomicLoad(&kmeans_bitmasks_high[3]));

        if(uniforms.partition_count == 2u) {
            let b0 = coverage_bitmaps_2[2 * i + 0];
            let b1 = coverage_bitmaps_2[2 * i + 1];

            let v1 = popcount64(xor64(a0, b0)) + popcount64(xor64(a1, b1));
            let v2 = popcount64(xor64(a0, b1)) + popcount64(xor64(a1, b0));

            mismatch_count = min(v1, v2) / 2u;
        }
        else if(uniforms.partition_count == 3u) {
            let b0 = coverage_bitmaps_3[3 * i + 0];
			let b1 = coverage_bitmaps_3[3 * i + 1];
			let b2 = coverage_bitmaps_3[3 * i + 2];

			let p00 = popcount64(xor64(a0, b0));
            let p01 = popcount64(xor64(a0, b1));
            let p02 = popcount64(xor64(a0, b2));
            let p10 = popcount64(xor64(a1, b0));
            let p11 = popcount64(xor64(a1, b1));
            let p12 = popcount64(xor64(a1, b2));
            let p20 = popcount64(xor64(a2, b0));
            let p21 = popcount64(xor64(a2, b1));
            let p22 = popcount64(xor64(a2, b2));

            let s0 = p11 + p22;
            let s1 = p12 + p21;
            let v0 = min(s0, s1) + p00;

            let s2 = p10 + p22;
            let s3 = p12 + p20;
            let v1 = min(s2, s3) + p01;

            let s4 = p10 + p21;
            let s5 = p11 + p20;
            let v2 = min(s4, s5) + p02;

            mismatch_count = min(v0, min(v1, v2)) / 2u;
		}
		else if(uniforms.partition_count == 4u) {
			let b0 = coverage_bitmaps_4[4 * i + 0];
			let b1 = coverage_bitmaps_4[4 * i + 1];
			let b2 = coverage_bitmaps_4[4 * i + 2];
			let b3 = coverage_bitmaps_4[4 * i + 3];

			let p00 = popcount64(xor64(a0, b0));
			let p01 = popcount64(xor64(a0, b1));
			let p02 = popcount64(xor64(a0, b2));
			let p03 = popcount64(xor64(a0, b3));
			let p10 = popcount64(xor64(a1, b0));
			let p11 = popcount64(xor64(a1, b1));
			let p12 = popcount64(xor64(a1, b2));
			let p13 = popcount64(xor64(a1, b3));
			let p20 = popcount64(xor64(a2, b0));
			let p21 = popcount64(xor64(a2, b1));
			let p22 = popcount64(xor64(a2, b2));
			let p23 = popcount64(xor64(a2, b3));
			let p30 = popcount64(xor64(a3, b0));
			let p31 = popcount64(xor64(a3, b1));
			let p32 = popcount64(xor64(a3, b2));
			let p33 = popcount64(xor64(a3, b3));

			let mx23 = min(p22 + p33, p23 + p32);
            let mx13 = min(p21 + p33, p23 + p31);
            let mx12 = min(p21 + p32, p22 + p31);
            let mx03 = min(p20 + p33, p23 + p30);
            let mx02 = min(p20 + p32, p22 + p30);
            let mx01 = min(p21 + p30, p20 + p31);

            let v0 = p00 + min(p11 + mx23, min(p12 + mx13, p13 + mx12));
            let v1 = p01 + min(p10 + mx23, min(p12 + mx03, p13 + mx02));
            let v2 = p02 + min(p11 + mx03, min(p10 + mx13, p13 + mx01));
            let v3 = p03 + min(p11 + mx02, min(p12 + mx01, p10 + mx12));

            mismatch_count = min(v0, min(v1, min(v2, v3))) / 2u;
        }

        let out_idx = block_idx * BLOCK_MAX_PARTITIONINGS * i;
        mismatch_counts[out_idx] = mismatch_count;
    }
}