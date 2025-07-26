#include <cstdint>
#include <vector>
#include <cstddef>
#include <array>
#include <assert.h>

#include <webgpu/webgpu.h>
#include <webgpu/webgpu_cpp.h>

#if defined(EMSCRIPTEN)
#include <emscripten.h>
#endif

const unsigned int BLOCK_MAX_TEXELS = 144;

const unsigned int BLOCK_MAX_COMPONENTS = 4;

const unsigned int BLOCK_MAX_PARTITIONS = 4;

const unsigned int BLOCK_MAX_PARTITIONINGS = 1024;

const unsigned int BLOCK_MAX_WEIGHTS = 64;

const unsigned int BLOCK_MAX_WEIGHTS_2PLANE = BLOCK_MAX_WEIGHTS / 2;

const unsigned int BLOCK_MIN_WEIGHT_BITS = 24;

const unsigned int BLOCK_MAX_WEIGHT_BITS = 96;

const unsigned int PARTITION_INDEX_BITS = 10;

const unsigned int WEIGHTS_MAX_BLOCK_MODES = 2048;

const unsigned int WEIGHTS_MAX_DECIMATION_MODES = 87;

const float WEIGHTS_TEXEL_SUM = 16.0f;

const uint16_t BLOCK_BAD_BLOCK_MODE = 0xFFFFu;

const unsigned int MAX_ANGULAR_QUANT = 7; //QUANT_12

const unsigned int SINCOS_STEPS = 1024;
const unsigned int ANGULAR_STEPS = 16;
const float PI = 3.14159265358979323846;

//channel error weights
const float ERROR_WEIGHT_R = 1.0f; //0.30f * 2.25f
const float ERROR_WEIGHT_G = 1.0f; //0.59f * 2.25f
const float ERROR_WEIGHT_B = 1.0f; //0.11f * 2.25f
const float ERROR_WEIGHT_A = 1.0f;

const unsigned int QUANT_LEVELS = 21; //QUANT_2 to QUANT_256

//relates to number if integers used by a given endpoint encoding. Used in shader passes 9, 10 and 11
const unsigned int NUM_INT_COUNTS = 4; //(2, 4, 6 and 8)
const unsigned int MAX_INT_COUNT_COMBINATIONS = 13; //4 partitions, int count can only differ by 1 step

const unsigned int TUNE_MAX_TRIAL_CANDIDATES = 8; //The maximum number of candidate encodings tested for each encoding mode

enum quant_method
{
	QUANT_2 = 0,
	QUANT_3 = 1,
	QUANT_4 = 2,
	QUANT_5 = 3,
	QUANT_6 = 4,
	QUANT_8 = 5,
	QUANT_10 = 6,
	QUANT_12 = 7,
	QUANT_16 = 8,
	QUANT_20 = 9,
	QUANT_24 = 10,
	QUANT_32 = 11,
	QUANT_40 = 12,
	QUANT_48 = 13,
	QUANT_64 = 14,
	QUANT_80 = 15,
	QUANT_96 = 16,
	QUANT_128 = 17,
	QUANT_160 = 18,
	QUANT_192 = 19,
	QUANT_256 = 20
};

/**
 * @brief The number of levels use by an ASTC quantization method.
 *
 * @param method   The quantization method
 *
 * @return   The number of levels used by @c method.
 */
static inline unsigned int get_quant_level(quant_method method)
{
	switch (method)
	{
	case QUANT_2:   return   2;
	case QUANT_3:   return   3;
	case QUANT_4:   return   4;
	case QUANT_5:   return   5;
	case QUANT_6:   return   6;
	case QUANT_8:   return   8;
	case QUANT_10:  return  10;
	case QUANT_12:  return  12;
	case QUANT_16:  return  16;
	case QUANT_20:  return  20;
	case QUANT_24:  return  24;
	case QUANT_32:  return  32;
	case QUANT_40:  return  40;
	case QUANT_48:  return  48;
	case QUANT_64:  return  64;
	case QUANT_80:  return  80;
	case QUANT_96:  return  96;
	case QUANT_128: return 128;
	case QUANT_160: return 160;
	case QUANT_192: return 192;
	case QUANT_256: return 256;
	}

	// Unreachable - the enum is fully described
	return 0;
}


unsigned int get_ise_sequence_bitcount(unsigned int character_count, quant_method quant_level);


//temporary constants
const int quant_limit = QUANT_12;
const int TUNE_MAX_ANGULAR_QUANT = 7; //QUANT_12

//number of angular steps for each quant level
const uint32_t steps_for_quant_level[] = {
	2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32
};



//Metadata struct definitions
//-----------------------------------------------------------------------------------------------------------------------------------

struct dt_init_working_buffers
{
	uint8_t weight_count_of_texel[BLOCK_MAX_TEXELS];
	uint8_t grid_weights_of_texel[BLOCK_MAX_TEXELS][4];
	uint8_t weights_of_texel[BLOCK_MAX_TEXELS][4];

	uint8_t texel_count_of_weight[BLOCK_MAX_WEIGHTS];
	uint8_t texels_of_weight[BLOCK_MAX_WEIGHTS][BLOCK_MAX_TEXELS];
	uint8_t texel_weights_of_weight[BLOCK_MAX_WEIGHTS][BLOCK_MAX_TEXELS];
};

/**
 * @brief This struct directly mirrors the DecimationInfo struct in WGSL.
 *
 * It contains all the metadata for a single decimation mode, including counts
 * and offsets into the larger packed data buffers. It uses fixed-size arrays
 * and uint32_t to ensure its memory layout is identical to the one expected
 * by the GPU.
 */
struct alignas(16) decimation_info {
	uint32_t texel_count;
	uint32_t weight_count;
	uint32_t weight_x;
	uint32_t weight_y;

	uint32_t max_quant_level;
	uint32_t max_angular_steps;
	uint32_t max_quant_steps;
	uint32_t _padding;

	std::array<uint32_t, BLOCK_MAX_TEXELS> texel_weight_count;
	std::array<uint32_t, BLOCK_MAX_TEXELS> texel_weights_offset;

	std::array<uint32_t, BLOCK_MAX_WEIGHTS> weight_texel_count;
	std::array<uint32_t, BLOCK_MAX_WEIGHTS> weight_texels_offset;
};

struct alignas(16) TexelToWeightMap {
	uint32_t weight_index;
	float contribution;

	uint32_t _padding1;
	uint32_t _padding2;
};

struct alignas(16) WeightToTexelMap {
	uint32_t texel_index;
	float contribution;

	uint32_t _padding1;
	uint32_t _padding2;
};

/**
 * @brief A container for the dynamically sized packed data arrays.
 *
 * All decimation modes will have their data appended to these vectors.
 */
struct packed_decimation_data {
	std::vector<TexelToWeightMap> texel_to_weight_map_data;

	std::vector<WeightToTexelMap> weight_to_texel_map_data;

};

struct alignas(16) decimation_mode {
	/** @brief The max weight precision for 1 plane, or -1 if not supported. */
	uint32_t maxprec_1plane;

	/** @brief The max weight precision for 2 planes, or -1 if not supported. */
	uint32_t maxprec_2planes;

	/**
	 * @brief Bitvector indicating weight quant modes used by active 1 plane block modes.
	 *
	 * Bit 0 = QUANT_2, Bit 1 = QUANT_3, etc.
	 */
	uint32_t refprec_1plane;

	/**
	 * @brief Bitvector indicating weight quant methods used by active 2 plane block modes.
	 *
	 * Bit 0 = QUANT_2, Bit 1 = QUANT_3, etc.
	 */
	uint32_t refprec_2planes;
};

/**
 * @brief struct for holding decoded info of block mode
 */
struct alignas(16) block_mode {
	uint32_t mode_index;

	uint32_t decimation_mode;

	uint32_t quant_mode;

	uint32_t weight_bits;

	uint32_t is_dual_plane : 1;

	uint32_t _padding1;
	uint32_t _padding2;
	uint32_t _padding3;
};

/**
 * @brief holds variables that are uniform for all blocks
 */
struct alignas(16) uniform_variables {
	uint32_t xdim;
	uint32_t ydim;

	uint32_t texel_count;

	uint32_t decimation_mode_count;
	uint32_t block_mode_count;

	uint32_t quant_limit;
	uint32_t partition_count;
	uint32_t tune_candidate_limit;

	float channel_weights[4];
};

/**
 * @brief holds the combined metadata used in the compression of a block
 */
struct block_descriptor {
	uniform_variables uniform_variables;

	decimation_info decimation_info_metadata[WEIGHTS_MAX_DECIMATION_MODES]; //decimation info is split into two parts for GPU memory optimisation
	packed_decimation_data decimation_info_packed;

	decimation_mode decimation_modes[WEIGHTS_MAX_DECIMATION_MODES];


	block_mode block_modes[WEIGHTS_MAX_BLOCK_MODES];

	uint32_t block_mode_index[WEIGHTS_MAX_BLOCK_MODES];

};

/**
 * @brief holds indices of a pair (block, decimation_mode).
 * Not all legal decimation modes will actually be viable for every block.
 * We only preform trials for viable (block,decimation_mode) pairs, to avoid doing unneccesary work.
 */
struct alignas(16) DecimationModeTrial {
	uint32_t block_index;
	uint32_t decimation_mode_index;

	uint32_t padding1;
	uint32_t padding2;
};

/**
 * @brief holds indices of a pair (block, block mode_mode), as well as the index of the deciamtion mode trial
 * corresponding to the block mode
 * We only preform trials for viable (block, block_mode) pairs, to avoid doing unneccesary work.
 */
struct alignas(16) BlockModeTrial {
	uint32_t block_index;
	uint32_t block_mode_index;
	uint32_t decimation_mode_trial_index;

	uint32_t padding1;
};


//End of metadata struct definitions
//-----------------------------------------------------------------------------------------------------------------------------------

static bool decode_block_mode_2d(
	unsigned int block_mode,
	unsigned int& x_weights,
	unsigned int& y_weights,
	bool& is_dual_plane,
	unsigned int& quant_mode,
	unsigned int& weight_bits
);

/**
 * @brief Prepares GPU data for a single decimation mode.
 *
 * This function calculates the decimation tables for one mode and appends the
 * results to the provided decimation info structs.
 *
 * @param mode_index The index of the mode we are currently processing.
 * @param x_texels The number of texels in the X dimension.
 * @param y_texels The number of texels in the Y dimension.
 * @param x_weights The number of weights in the X dimension.
 * @param y_weights The number of weights in the Y dimension.
 * @param[out] out_data The destination data set to populate.
 * @param wb A temporary working buffer.
 */
static void init_decimation_info(
	unsigned int x_texels,
	unsigned int y_texels,
	unsigned int x_weights,
	unsigned int y_weights,
	decimation_info& out_data,
	packed_decimation_data& out_data_packed,
	dt_init_working_buffers& wb
);

/**
 * @brief Allocate a single 2D decimation table entry.
 *
 * @param x_texels          The number of texels in the X dimension.
 * @param y_texels          The number of texels in the Y dimension.
 * @param x_weights         The number of weights in the X dimension.
 * @param y_weights         The number of weights in the Y dimension.
 * @param block_descriptor  The block descriptor we are populating.
 * @param wb                The decimation table init scratch working buffers.
 * @param index             The packed array index to populate.
 */
static void construct_dt_entry(
	unsigned int x_texels,
	unsigned int y_texels,
	unsigned int x_weights,
	unsigned int y_weights,
	block_descriptor& block_descriptor,
	dt_init_working_buffers& wb,
	unsigned int index
);


/**
 * @brief Construct the structures containing precomputed metadata used in compression
 */
void construct_metadata_structures(
	unsigned int x_texels,
	unsigned int y_texels,
	block_descriptor& block_descriptor
);


/**
 * @brief Construct sin and cos tables for use in shaders
 */
void construct_angular_tables(
	std::vector<float>& sinTable,
	std::vector<float>& cosTable
);

//Block data struct definitions
//-----------------------------------------------------------------------------------------------------------------------------------

//inputBlocksBuffer structs
struct alignas(16) Pixel {
	float data[4];
	uint32_t partition;

	uint32_t padding1;
	uint32_t padding2;
	uint32_t padding3;
};

struct alignas(16) InputBlock {
	Pixel pixels[BLOCK_MAX_TEXELS];
	uint32_t partition_pixel_counts[BLOCK_MAX_PARTITIONS];
};

//pass1_output_idealEndpointsAndWeights structs
//per partitoin data of IdealEndpointsAndWeights
struct alignas(16) IdealEndpointsAndWeights_p {
	float avg[4];
	float dir[4];
	float endpoint0[4];
	float endpoint1[4];
};

//output struct of the ideal endpitnts and weights shader
struct alignas(16) IdealEndpointsAndWeights {
	IdealEndpointsAndWeights_p partitions[4];
	float weights[BLOCK_MAX_TEXELS];

	float weight_error_scale[BLOCK_MAX_TEXELS];

	uint32_t is_constant_weight_error_scale;
	float min_weight_cutoff;

	uint32_t _padding1;
	uint32_t _padding2;
};

//output struct of lowest and highest weights shader
struct alignas(16) HighestAndLowestWeight {
	float lowest_weight;
	int32_t weight_span;
	float error;
	float cut_low_error;
	float cut_high_error;

	uint32_t _padding1;
	uint32_t _padding2;
	uint32_t _padding3;
};

struct alignas(16) FinalValueRange {
	float low;
	float high;

	uint32_t _padding1;
	uint32_t _padding2;
};

//output of weights and error for block mode shader
struct alignas(16) QuantizationResult {
	float error;
	int32_t bitcount;

	uint32_t _padding1;
	uint32_t _padding2;

	uint32_t quantized_weights[BLOCK_MAX_WEIGHTS];
};

//output of encoding choice errors shader
struct alignas(16) EncodingChoiceErrors {
	float rgb_scale_error;
	float rgb_luma_error;
	float luminance_error;
	float alpha_drop_error;

	uint32_t can_offset_encode;
	uint32_t can_blue_contract;

	uint32_t _padding1;
	uint32_t _padding2;
};

//output of color endpoint combinations shader
struct alignas(16) CombinedEndpointFormats {
	float error;

	uint32_t _padding0;
	uint32_t _padding1;
	uint32_t _padding2;

	uint32_t formats[4];
};

//output of best endpoint combinations shader
struct alignas(16) ColorCombinationResult {
	float total_error;

	uint32_t best_quant_level;
	uint32_t best_quant_level_mod;

	uint32_t _padding1;

	uint32_t best_ep_formats[4];
};

//output of final candidates shader
struct alignas(16) FinalCandidate {
	uint32_t block_mode_index;
	uint32_t block_mode_trial_index;
	float total_error;
	uint32_t quant_level;
	uint32_t quant_level_mod;

	uint32_t _padding1;
	uint32_t _padding2;
	uint32_t _padding3;

	uint32_t formats[4];
};


//End of block data struct definitions
//-----------------------------------------------------------------------------------------------------------------------------------

/**
 * @brief Split image into a vector of blocks containing raw color data
 */
std::vector<InputBlock> SplitImageIntoBlocks(
	uint8_t* imageData,
	int width,
	int height,
	int blockWidth,
	int blockHeight
);



class ASTCEncoder {
public:
	ASTCEncoder(wgpu::Device device, uint32_t textureWidth, uint32_t textureHeight, uint8_t blockXDim, uint8_t blockYDim);

	~ASTCEncoder();

	void encode(uint8_t* imageData);

private:
	void initMetadata();
	void initBindGroupLayouts();
	void initBuffers();
	void initPipelines();
	void initBindGroups();

	wgpu::Device device;
	wgpu::Queue queue;

	block_descriptor block_descriptor; //contains metadata used in compression

	std::vector<float> sin_table; //precomputed sine values
	std::vector<float> cos_table; //precomputed cosine values

	uint32_t numBlocks;
	uint32_t blocksX;
	uint32_t blocksY;
	uint32_t textureWidth;
	uint32_t textureHeight;
	uint8_t blockXDim;
	uint8_t blockYDim;

	std::vector<DecimationModeTrial> decimation_mode_trials; //(block, deciamtion mode) pairs to test
	std::vector<BlockModeTrial> block_mode_trials; //(block, block mode) pairs to test
	std::vector<uint32_t> modes_per_block;// number of block modes for each block
	std::vector<uint32_t> block_mode_trial_offsets; //global index of first block mode trial for each block

	//Shader modules 
	wgpu::ShaderModule pass1_idealEndpointsShader;
	wgpu::ShaderModule pass2_decimatedWeightsShader;
	wgpu::ShaderModule pass3_angularOffsetsShader;
	wgpu::ShaderModule pass4_lowestAndHighestWeightShader;
	wgpu::ShaderModule pass5_valuesForQuantLevelsShader;
	wgpu::ShaderModule pass6_remapLowAndHighValuesShader;
	wgpu::ShaderModule pass7_weightsAndErrorForBMShader;
	wgpu::ShaderModule pass8_encodingChoiceErrorsShader;
	wgpu::ShaderModule pass9_computeColorErrorShader;
	wgpu::ShaderModule pass10_colorEndpointCombinationsShader_2part;
	wgpu::ShaderModule pass10_colorEndpointCombinationsShader_3part;
	wgpu::ShaderModule pass10_colorEndpointCombinationsShader_4part;
	wgpu::ShaderModule pass11_bestEndpointCombinationsForModeShader_1part;
	wgpu::ShaderModule pass11_bestEndpointCombinationsForModeShader_2part;
	wgpu::ShaderModule pass11_bestEndpointCombinationsForModeShader_3part;
	wgpu::ShaderModule pass11_bestEndpointCombinationsForModeShader_4part;
	wgpu::ShaderModule pass12_findTopNCandidatesShader;


	//Compute Pipelines
	wgpu::ComputePipeline pass1_pipeline;
	wgpu::ComputePipeline pass2_pipeline;
	wgpu::ComputePipeline pass3_pipeline;
	wgpu::ComputePipeline pass4_pipeline;
	wgpu::ComputePipeline pass5_pipeline;
	wgpu::ComputePipeline pass6_pipeline;
	wgpu::ComputePipeline pass7_pipeline;
	wgpu::ComputePipeline pass8_pipeline;
	wgpu::ComputePipeline pass9_pipeline;
	wgpu::ComputePipeline pass10_pipeline_2part;
	wgpu::ComputePipeline pass10_pipeline_3part;
	wgpu::ComputePipeline pass10_pipeline_4part;
	wgpu::ComputePipeline pass11_pipeline_1part;
	wgpu::ComputePipeline pass11_pipeline_2part;
	wgpu::ComputePipeline pass11_pipeline_3part;
	wgpu::ComputePipeline pass11_pipeline_4part;
	wgpu::ComputePipeline pass12_pipeline;	

	//Bind Group Layouts
	wgpu::BindGroupLayout pass1_bindGroupLayout;
	wgpu::BindGroupLayout pass2_bindGroupLayout;
	wgpu::BindGroupLayout pass3_bindGroupLayout;
	wgpu::BindGroupLayout pass4_bindGroupLayout;
	wgpu::BindGroupLayout pass5_bindGroupLayout;
	wgpu::BindGroupLayout pass6_bindGroupLayout;
	wgpu::BindGroupLayout pass7_bindGroupLayout;
	wgpu::BindGroupLayout pass8_bindGroupLayout;
	wgpu::BindGroupLayout pass9_bindGroupLayout;
	wgpu::BindGroupLayout pass10_bindGroupLayout;
	wgpu::BindGroupLayout pass11_bindGroupLayout_1part;
	wgpu::BindGroupLayout pass11_bindGroupLayout_234part;
	wgpu::BindGroupLayout pass12_bindGroupLayout;

	//Buffers
	wgpu::Buffer uniformsBuffer;

	wgpu::Buffer decimationModeTrialsBuffer;
	wgpu::Buffer blockModeTrialsBuffer;
	wgpu::Buffer modesPerBlockBuffer;
	wgpu::Buffer blockModeTrialOffsetsBuffer;

	//Buffers for block mode info (constant after setup)
	wgpu::Buffer blockModesBuffer;
	wgpu::Buffer blockModeIndexBuffer;
	wgpu::Buffer decimationModesBuffer;
	wgpu::Buffer decimationInfoBuffer;
	wgpu::Buffer texelToWeightMapBuffer;
	wgpu::Buffer weightToTexelMapBuffer;

	//Buffers for storing sin and cos function values
	wgpu::Buffer sinBuffer;
	wgpu::Buffer cosBuffer;

	//Block data buffers (they contain the data for individual blocks)
	wgpu::Buffer inputBlocksBuffer;
	wgpu::Buffer pass1_output_idealEndpointsAndWeights;
	wgpu::Buffer pass2_output_decimatedWeights;
	wgpu::Buffer pass3_output_angular_offsets;
	wgpu::Buffer pass4_output_lowestAndHighestWeight;
	wgpu::Buffer pass5_output_lowValues;
	wgpu::Buffer pass5_output_highValues;
	wgpu::Buffer pass6_output_finalValueRanges;
	wgpu::Buffer pass7_output_quantizationResults;
	wgpu::Buffer pass8_output_encodingChoiceErrors;
	wgpu::Buffer pass9_output_colorFormatErrors;
	wgpu::Buffer pass9_output_colorFormats;
	wgpu::Buffer pass10_output_colorEndpointCombinations;
	wgpu::Buffer pass11_output_bestEndpointCombinationsForMode;
	wgpu::Buffer pass12_output_finalCandidates;

	//Readback buffers
	wgpu::Buffer readbackBuffer;
	wgpu::Buffer readbackBuffer2;
	wgpu::Buffer readbackBuffer6_1;
	wgpu::Buffer readbackBuffer6_2;
	wgpu::Buffer readbackBuffer7;
	wgpu::Buffer readbackBuffer10;
	wgpu::Buffer readbackBuffer11;
	wgpu::Buffer readbackBuffer12;

	//Bind Groups
	wgpu::BindGroup pass1_bindGroup;
	wgpu::BindGroup pass2_bindGroup;
	wgpu::BindGroup pass3_bindGroup;
	wgpu::BindGroup pass4_bindGroup;
	wgpu::BindGroup pass5_bindGroup;
	wgpu::BindGroup pass6_bindGroup;
	wgpu::BindGroup pass7_bindGroup;
	wgpu::BindGroup pass8_bindGroup;
	wgpu::BindGroup pass9_bindGroup;
	wgpu::BindGroup pass10_bindGroup;
	wgpu::BindGroup pass11_bindGroup_1part;
	wgpu::BindGroup pass11_bindGroup_234part;
	wgpu::BindGroup pass12_bindGroup;
};