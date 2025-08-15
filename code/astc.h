#pragma once

#include <cstdint>
#include <vector>
#include <cstddef>
#include <array>
#include <assert.h>
#include <atomic>
#include <functional>

#include <webgpu/webgpu.h>
#include <webgpu/webgpu_cpp.h>

#if defined(EMSCRIPTEN)
#include <emscripten.h>
#endif

const unsigned int BLOCK_MAX_TEXELS = 144;

const unsigned int BLOCK_MAX_COMPONENTS = 4;

const unsigned int BLOCK_MAX_PARTITIONS = 4;

const unsigned int BLOCK_MAX_PARTITIONINGS = 1024;

const unsigned int TUNE_MAX_PARTITIONING_CANDIDATES = 2;

const unsigned int BLOCK_MAX_WEIGHTS = 64;

const unsigned int BLOCK_MAX_WEIGHTS_2PLANE = BLOCK_MAX_WEIGHTS / 2;
const unsigned int WEIGHTS_PLANE2_OFFSET = BLOCK_MAX_WEIGHTS_2PLANE;

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
//const float ERROR_WEIGHT_R = 1.0f; //0.30f * 2.25f
//const float ERROR_WEIGHT_G = 1.0f; //0.59f * 2.25f
//const float ERROR_WEIGHT_B = 1.0f; //0.11f * 2.25f
//const float ERROR_WEIGHT_A = 1.0f;

const float ERROR_WEIGHT_R = 0.30f * 2.25f;
const float ERROR_WEIGHT_G = 0.59f * 2.25f;
const float ERROR_WEIGHT_B = 0.11f * 2.25f;
const float ERROR_WEIGHT_A = 1.0f;

const unsigned int QUANT_LEVELS = 21; //QUANT_2 to QUANT_256

//relates to number if integers used by a given endpoint encoding. Used in shader passes 9, 10 and 11
const unsigned int NUM_INT_COUNTS = 4; //(2, 4, 6 and 8)
const unsigned int MAX_INT_COUNT_COMBINATIONS = 13; //4 partitions, int count can only differ by 1 step

const unsigned int TUNE_MAX_TRIAL_CANDIDATES = 4; //The maximum number of candidate encodings tested for each encoding mode

//The maximum number of texels used during partition selection for texel clustering
const unsigned int BLOCK_MAX_KMEANS_TEXELS = 64;

const float ERROR_CALC_DEFAULT = 1e30f;

static constexpr uint16_t BLOCK_BAD_PARTITIONING = 0xFFFFu;

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

struct alignas(16) PackedBlockModeLookup {
	uint32_t block_mode_index;
	uint32_t decimation_mode_lookup_idx; //index of corresponding decimation mode trial

	uint32_t _padding1;
	uint32_t _padding2;
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

	uint32_t valid_decimation_mode_count;
	uint32_t valid_block_mode_count;

	uint32_t quant_limit;
	uint32_t partition_count;
	uint32_t tune_candidate_limit;

	uint32_t _padding1;
	uint32_t _padding2;

	float channel_weights[4];
};

struct partition_info {
	uint16_t partition_count;
	uint16_t partition_index;
	uint8_t partition_texel_count[BLOCK_MAX_PARTITIONS];
	uint8_t partition_of_texel[BLOCK_MAX_TEXELS];
	uint8_t texels_of_partition[BLOCK_MAX_PARTITIONS][BLOCK_MAX_TEXELS];
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


	partition_info partitionings[(3 * BLOCK_MAX_PARTITIONINGS) + 1];
	uint16_t partitioning_packed_index[3][BLOCK_MAX_PARTITIONINGS];

	/** @brief The active texels for k-means partition selection. */
	uint8_t kmeans_texels[BLOCK_MAX_KMEANS_TEXELS];

	/** @brief The number of selected partitionings for 1/2/3/4 partitionings. */
	unsigned int partitioning_count_selected[BLOCK_MAX_PARTITIONS];

	/** @brief The number of partitionings for 1/2/3/4 partitionings. */
	unsigned int partitioning_count_all[BLOCK_MAX_PARTITIONS];

	/**
	 * @brief The canonical partition coverage pattern used during block partition search.
	 *
	 * Indexed by remapped index, not physical index.
	 */
	uint64_t coverage_bitmaps_2[BLOCK_MAX_PARTITIONINGS][2];
	uint64_t coverage_bitmaps_3[BLOCK_MAX_PARTITIONINGS][3];
	uint64_t coverage_bitmaps_4[BLOCK_MAX_PARTITIONINGS][4];

	/**
	 * @brief Get the partition info table for a given partition count.
	 *
	 * @param partition_count   The number of partitions we want the table for.
	 *
	 * @return The pointer to the table of 1024 entries (for 2/3/4 parts) or 1 entry (for 1 part).
	 */
	const partition_info* get_partition_table(unsigned int partition_count) const
	{
		if (partition_count == 1)
		{
			partition_count = 5;
		}
		unsigned int index = (partition_count - 2) * BLOCK_MAX_PARTITIONINGS;
		return this->partitionings + index;
	}

	/**
	 * @brief Get the partition info structure for a given partition count and seed.
	 *
	 * @param partition_count   The number of partitions we want the info for.
	 * @param index             The partition seed (between 0 and 1023).
	 *
	 * @return The partition info structure.
	 */
	const partition_info& get_partition_info(unsigned int partition_count, unsigned int index) const
	{
		unsigned int packed_index = 0;
		if (partition_count >= 2)
		{
			packed_index = this->partitioning_packed_index[partition_count - 2][index];
		}

		assert(packed_index != BLOCK_BAD_PARTITIONING && packed_index < this->partitioning_count_all[partition_count - 1]);
		auto& result = get_partition_table(partition_count)[packed_index];
		assert(index == result.partition_index);
		return result;
	}

	/**
	 * @brief Get the partition info structure for a given partition count and seed.
	 *
	 * @param partition_count   The number of partitions we want the info for.
	 * @param packed_index      The raw array offset.
	 *
	 * @return The partition info structure.
	 */
	const partition_info& get_raw_partition_info(unsigned int partition_count, unsigned int packed_index) const
	{
		assert(packed_index != BLOCK_BAD_PARTITIONING && packed_index < this->partitioning_count_all[partition_count - 1]);
		auto& result = get_partition_table(partition_count)[packed_index];
		return result;
	}
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
 * @brief Populate the partition tables for the target block size.
 */
void init_partition_tables(
	block_descriptor& block_descriptor,
	bool can_omit_partitionings,
	unsigned int partition_count_cutoff
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
	float data_min[4];
	float data_max[4];

	uint32_t grayscale;
	uint32_t partitioning_idx;
	uint32_t xpos;
	uint32_t ypos;
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
	
	uint32_t color_formats_matched;
	uint32_t final_quant_mode; // The quant mode after checking the mod version

	uint32_t formats[4];

	uint32_t quantized_weights[BLOCK_MAX_WEIGHTS];
	IdealEndpointsAndWeights_p candidate_partitions[4];

	uint32_t final_formats[4];
	uint32_t packed_color_values[32]; //8 integers per partition, 4 partitions
};

//output of unpack color endpoints shader
struct alignas(16) UnpackedEndpoints {
	int32_t endpoint0[4][4];
	int32_t endpoint1[4][4];
};

struct alignas(16) SymbolicBlock {
	float errorval;

	uint32_t block_mode_index;
	uint32_t partition_count;
	uint32_t partition_index;

	uint32_t partition_formats_matched;
	uint32_t quant_mode;

	uint32_t _padding1;
	uint32_t _padding2;

	uint32_t partition_formats[4];
	uint32_t packed_color_values[32];
	uint32_t quantized_weights[BLOCK_MAX_WEIGHTS];
};


//End of block data struct definitions
//-----------------------------------------------------------------------------------------------------------------------------------


//Utility functions
//-----------------------------------------------------------------------------------------------------------------------------------

/**
 * @brief Return the number of bits needed to encode an ISE sequence.
 *
 * This implementation assumes that the @c quant level is untrusted, given it may come from random
 * data being decompressed, so we return an arbitrary unencodable size if that is the case.
 *
 * @param character_count   The number of items in the sequence.
 * @param quant_level       The desired quantization level.
 *
 * @return The number of bits needed to encode the BISE string.
 */
unsigned int get_ise_sequence_bitcount(unsigned int character_count, quant_method quant_level);

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

unsigned int find_best_partition_candidates(
	const block_descriptor& block_descriptor,
	const InputBlock& blk,
	unsigned int partition_count,
	unsigned int partition_search_limit,
	unsigned int best_partitions[TUNE_MAX_PARTITIONING_CANDIDATES],
	unsigned int requested_candidates
);

/**
 * @brief Encode a packed string using BISE.
 *
 * Note that BISE can return strings that are not a whole number of bytes in length, and ASTC can
 * start storing strings in a block at arbitrary bit offsets in the encoded data.
 *
 * @param         quant_level       The BISE alphabet size.
 * @param         character_count   The number of characters in the string.
 * @param         input_data        The unpacked string, one byte per character.
 * @param[in,out] output_data       The output packed string.
 * @param         bit_offset        The starting offset in the output storage.
 */
void encode_ise(
	quant_method quant_level,
	unsigned int character_count,
	const uint8_t* input_data,
	uint8_t* output_data,
	unsigned int bit_offset
);

void symbolic_to_physical(
	const block_descriptor& block_descriptor,
	const SymbolicBlock& symbolic_compressed_block,
	uint8_t physical_compressed_block[16]
);

class ASTCEncoder {
public:
	ASTCEncoder(const wgpu::Device& device);

	~ASTCEncoder();

	void init();
	void secondaryInit(uint32_t textureWidth, uint32_t textureHeight, uint8_t blockXDim, uint8_t blockYDim);

	void encode(uint8_t* imageData, uint8_t* dataOut, size_t dataLen);

	uint32_t numBlocks;
	uint32_t blocksX;
	uint32_t blocksY;

	const uint32_t batchSize = 512;

#if defined(EMSCRIPTEN)
	std::atomic<int> m_pending_pipelines;
	void initAsync(std::function<void()> on_initialized);
#endif

	bool is_initialized = false;

private:

	void initMetadata();
	void initTrialModes();
	void initBindGroupLayouts();
	void initBuffers();
	void initPipelines();
	void initBindGroups();
	void releasePerImageResources();

#if defined(EMSCRIPTEN)
	void initPipelinesAsync(std::function<void()> on_all_pipelines_created);
#endif

	wgpu::Device device;
	wgpu::Queue queue;

	block_descriptor block_descriptor; //contains metadata used in compression

	std::vector<float> sin_table; //precomputed sine values
	std::vector<float> cos_table; //precomputed cosine values

	uint32_t textureWidth;
	uint32_t textureHeight;
	uint8_t blockXDim;
	uint8_t blockYDim;

	std::vector<uint32_t> valid_decimation_modes; //Decimation modes that we actually consider for encoding
	std::vector<PackedBlockModeLookup> valid_block_modes; //Block modes that we actually consider for encoding

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
	wgpu::ShaderModule pass13_recomputeIdealEndpointsShader;
	wgpu::ShaderModule pass14_packColorEndpointsShader;
	wgpu::ShaderModule pass15_unpackColorEndpointsShader;
	wgpu::ShaderModule pass16_realignWeightsShader;
	wgpu::ShaderModule pass17_computeFinalErrorShader;
	wgpu::ShaderModule pass18_pickBestCandidateShader;

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
	wgpu::ComputePipeline pass13_pipeline;
	wgpu::ComputePipeline pass14_pipeline;
	wgpu::ComputePipeline pass15_pipeline;
	wgpu::ComputePipeline pass16_pipeline;
	wgpu::ComputePipeline pass17_pipeline;
	wgpu::ComputePipeline pass18_pipeline;

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
	wgpu::BindGroupLayout pass13_bindGroupLayout;
	wgpu::BindGroupLayout pass14_bindGroupLayout;
	wgpu::BindGroupLayout pass15_bindGroupLayout;
	wgpu::BindGroupLayout pass16_bindGroupLayout;
	wgpu::BindGroupLayout pass17_bindGroupLayout;
	wgpu::BindGroupLayout pass18_bindGroupLayout;

	//Buffers
	wgpu::Buffer uniformsBuffer;

	//Buffers for block mode info (constant after setup)
	wgpu::Buffer blockModesBuffer;
	wgpu::Buffer blockModeIndexBuffer;
	wgpu::Buffer decimationModesBuffer;
	wgpu::Buffer decimationInfoBuffer;
	wgpu::Buffer texelToWeightMapBuffer;
	wgpu::Buffer weightToTexelMapBuffer;

	wgpu::Buffer validDecimationModesBuffer;
	wgpu::Buffer validBlockModesBuffer;

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
	wgpu::Buffer pass12_output_topCandidates;
	wgpu::Buffer pass13_output_rgbsVectors;
	wgpu::Buffer pass15_output_unpackedEndpoints;
	wgpu::Buffer pass18_output_symbolicBlocks;

	wgpu::Buffer outputReadbackBuffer;

	wgpu::Buffer pass1111ReadbackBuffer;

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
	wgpu::BindGroup pass13_bindGroup;
	wgpu::BindGroup pass14_bindGroup;
	wgpu::BindGroup pass15_bindGroup;
	wgpu::BindGroup pass16_bindGroup;
	wgpu::BindGroup pass17_bindGroup;
	wgpu::BindGroup pass18_bindGroup;
};