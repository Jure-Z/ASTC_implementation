#include "astc.h"


static bool decode_block_mode_2d(
	unsigned int block_mode,
	unsigned int& x_weights,
	unsigned int& y_weights,
	bool& is_dual_plane,
	unsigned int& quant_mode,
	unsigned int& weight_bits
) {
	unsigned int base_quant_mode = (block_mode >> 4) & 1;
	unsigned int H = (block_mode >> 9) & 1;
	unsigned int D = (block_mode >> 10) & 1;
	unsigned int A = (block_mode >> 5) & 0x3;

	x_weights = 0;
	y_weights = 0;

	if ((block_mode & 3) != 0)
	{
		base_quant_mode |= (block_mode & 3) << 1;
		unsigned int B = (block_mode >> 7) & 3;
		switch ((block_mode >> 2) & 3)
		{
		case 0:
			x_weights = B + 4;
			y_weights = A + 2;
			break;
		case 1:
			x_weights = B + 8;
			y_weights = A + 2;
			break;
		case 2:
			x_weights = A + 2;
			y_weights = B + 8;
			break;
		case 3:
			B &= 1;
			if (block_mode & 0x100)
			{
				x_weights = B + 2;
				y_weights = A + 2;
			}
			else
			{
				x_weights = A + 2;
				y_weights = B + 6;
			}
			break;
		}
	}
	else
	{
		base_quant_mode |= ((block_mode >> 2) & 3) << 1;
		if (((block_mode >> 2) & 3) == 0)
		{
			return false;
		}

		unsigned int B = (block_mode >> 9) & 3;
		switch ((block_mode >> 7) & 3)
		{
		case 0:
			x_weights = 12;
			y_weights = A + 2;
			break;
		case 1:
			x_weights = A + 2;
			y_weights = 12;
			break;
		case 2:
			x_weights = A + 6;
			y_weights = B + 6;
			D = 0;
			H = 0;
			break;
		case 3:
			switch ((block_mode >> 5) & 3)
			{
			case 0:
				x_weights = 6;
				y_weights = 10;
				break;
			case 1:
				x_weights = 10;
				y_weights = 6;
				break;
			case 2:
			case 3:
				return false;
			}
			break;
		}
	}

	unsigned int weight_count = x_weights * y_weights * (D + 1);
	quant_mode = (base_quant_mode - 2) + 6 * H;
	is_dual_plane = D != 0;

	weight_bits = get_ise_sequence_bitcount(weight_count, static_cast<quant_method>(quant_mode));
	return (weight_count <= BLOCK_MAX_WEIGHTS &&
		weight_bits >= BLOCK_MIN_WEIGHT_BITS &&
		weight_bits <= BLOCK_MAX_WEIGHT_BITS);
}


 static void init_decimation_info (
	 unsigned int x_texels,
	 unsigned int y_texels,
	 unsigned int x_weights,
	 unsigned int y_weights,
	 decimation_info& out_data,
	 packed_decimation_data& out_data_packed,
	 dt_init_working_buffers& wb
 ) {
	 unsigned int texels_per_block = x_texels * y_texels;
	 unsigned int weights_per_block = x_weights * y_weights;

	 // PASS 1: Calculate relationships in temporary buffers (same as before).
	 // ... This logic remains unchanged ...
	 // Clear working buffers
	 for (unsigned int i = 0; i < weights_per_block; i++) { wb.texel_count_of_weight[i] = 0; }
	 for (unsigned int i = 0; i < texels_per_block; i++) { wb.weight_count_of_texel[i] = 0; }

	 for (unsigned int y = 0; y < y_texels; y++) {
		 for (unsigned int x = 0; x < x_texels; x++) {
			 unsigned int texel = y * x_texels + x;
			 unsigned int x_weight = (((1024 + x_texels / 2) / (x_texels - 1)) * x * (x_weights - 1) + 32) >> 6;
			 unsigned int y_weight = (((1024 + y_texels / 2) / (y_texels - 1)) * y * (y_weights - 1) + 32) >> 6;
			 unsigned int x_weight_frac = x_weight & 0xF;
			 unsigned int y_weight_frac = y_weight & 0xF;
			 unsigned int x_weight_int = x_weight >> 4;
			 unsigned int y_weight_int = y_weight >> 4;

			 unsigned int qweight[4];
			 qweight[0] = x_weight_int + y_weight_int * x_weights;
			 qweight[1] = qweight[0] + 1;
			 qweight[2] = qweight[0] + x_weights;
			 qweight[3] = qweight[2] + 1;

			 unsigned int prod = x_weight_frac * y_weight_frac;
			 unsigned int weight[4];
			 weight[3] = (prod + 8) >> 4;
			 weight[1] = x_weight_frac - weight[3];
			 weight[2] = y_weight_frac - weight[3];
			 weight[0] = 16 - x_weight_frac - y_weight_frac + weight[3];

			 for (unsigned int i = 0; i < 4; i++) {
				 if (weight[i] != 0) {
					 uint8_t texel_idx = wb.weight_count_of_texel[texel]++;
					 wb.grid_weights_of_texel[texel][texel_idx] = static_cast<uint8_t>(qweight[i]);
					 wb.weights_of_texel[texel][texel_idx] = static_cast<uint8_t>(weight[i]);
					 uint8_t weight_idx = wb.texel_count_of_weight[qweight[i]]++;
					 wb.texels_of_weight[qweight[i]][weight_idx] = static_cast<uint8_t>(texel);
					 wb.texel_weights_of_weight[qweight[i]][weight_idx] = static_cast<uint8_t>(weight[i]);
				 }
			 }
		 }
	 }


	 // PASS 2: Pack the data into the final GPU-friendly structures.
	 auto& meta = out_data;
	 auto& packed = out_data_packed;

	 // --- Pack texel-centric data ---
	 meta.texel_weights_offset[0] = static_cast<uint32_t>(packed.texel_to_weight_map_data.size());
	 for (unsigned int i = 0; i < texels_per_block; ++i) {
		 meta.texel_weight_count[i] = wb.weight_count_of_texel[i];

		 if (i > 0) {
			 meta.texel_weights_offset[i] = meta.texel_weights_offset[i - 1] + meta.texel_weight_count[i - 1];
		 }

		 for (unsigned int j = 0; j < wb.weight_count_of_texel[i]; ++j) {
			 uint32_t weight_index = wb.grid_weights_of_texel[i][j];
			 float contrib = static_cast<float>(wb.weights_of_texel[i][j]) * (1.0f / 16.0f);

			 packed.texel_to_weight_map_data.push_back({ weight_index, contrib });
		 }
	 }


	 // --- Pack weight-centric data ---
	 meta.weight_texels_offset[0] = static_cast<uint32_t>(packed.weight_to_texel_map_data.size());
	 for (unsigned int i = 0; i < weights_per_block; ++i) {
		 meta.weight_texel_count[i] = wb.texel_count_of_weight[i];

		 if (i > 0) {
			 meta.weight_texels_offset[i] = meta.weight_texels_offset[i - 1] + meta.weight_texel_count[i - 1];
		 }

		 for (unsigned int j = 0; j < wb.texel_count_of_weight[i]; ++j) {
			 uint32_t texel_index = static_cast<uint32_t>(wb.texels_of_weight[i][j]);
			 float contrib = static_cast<float>(wb.texel_weights_of_weight[i][j]) * (1.0f / 16.0f);

			 packed.weight_to_texel_map_data.push_back({ texel_index, contrib });
		 }
	 }

	 // --- Store final metadata for this mode ---
	 meta.texel_count = texels_per_block;
	 meta.weight_count = weights_per_block;
	 meta.weight_x = x_weights;
	 meta.weight_y = y_weights;
 }


 static void construct_dt_entry(
	 unsigned int x_texels,
	 unsigned int y_texels,
	 unsigned int x_weights,
	 unsigned int y_weights,
	 block_descriptor& block_descriptor,
	 dt_init_working_buffers& wb,
	 unsigned int index
 ) {
	 unsigned int weight_count = x_weights * y_weights;
	 assert(weight_count <= BLOCK_MAX_WEIGHTS);

	 bool try_2planes = (2 * weight_count) <= BLOCK_MAX_WEIGHTS;

	 decimation_info& decimation_info_metadata = block_descriptor.decimation_info_metadata[index];
	 packed_decimation_data& decimation_info_packed = block_descriptor.decimation_info_packed;

	 init_decimation_info(x_texels, y_texels, x_weights, y_weights, decimation_info_metadata, decimation_info_packed, wb);


	 int maxprec_1plane = -1;
	 int maxprec_2planes = -1;
	 for (int i = 0; i < 12; i++)
	 {
		 unsigned int bits_1plane = get_ise_sequence_bitcount(weight_count, static_cast<quant_method>(i));
		 if (bits_1plane >= BLOCK_MIN_WEIGHT_BITS && bits_1plane <= BLOCK_MAX_WEIGHT_BITS)
		 {
			 maxprec_1plane = i;
		 }

		 if (try_2planes)
		 {
			 unsigned int bits_2planes = get_ise_sequence_bitcount(2 * weight_count, static_cast<quant_method>(i));
			 if (bits_2planes >= BLOCK_MIN_WEIGHT_BITS && bits_2planes <= BLOCK_MAX_WEIGHT_BITS)
			 {
				 maxprec_2planes = i;
			 }
		 }
	 }

	 // At least one of the two should be valid ...
	 assert(maxprec_1plane >= 0 || maxprec_2planes >= 0);
	 block_descriptor.decimation_modes[index].maxprec_1plane = static_cast<int8_t>(maxprec_1plane);
	 block_descriptor.decimation_modes[index].maxprec_2planes = static_cast<int8_t>(maxprec_2planes);
	 block_descriptor.decimation_modes[index].refprec_1plane = 0;
	 block_descriptor.decimation_modes[index].refprec_2planes = 0;

	 //temporary solution
	 //precompute max_angular_steps for decimation mode
	 int max_weight_quant = std::min(static_cast<int>(QUANT_32), quant_limit);
	 int max_precision = std::min((int)block_descriptor.decimation_modes[index].maxprec_1plane, TUNE_MAX_ANGULAR_QUANT);
	 max_precision = std::min(max_precision, max_weight_quant);

	 uint32_t max_angular_steps = steps_for_quant_level[max_precision];

	 decimation_info_metadata.max_quant_level = max_precision;
	 decimation_info_metadata.max_angular_steps = max_angular_steps;
	 decimation_info_metadata.max_quant_steps = max_angular_steps;
 }

void construct_metadata_structures(
	 unsigned int x_texels,
	 unsigned int y_texels,
	 block_descriptor& block_descriptor
) {
	// Store a remap table for storing packed decimation modes.
	// Indexing uses [Y * 16 + X] and max size for each axis is 12.
	static const unsigned int MAX_DMI = 12 * 16 + 12;
	int decimation_mode_index[MAX_DMI];

	dt_init_working_buffers* wb = new dt_init_working_buffers;

	block_descriptor.uniform_variables.xdim = static_cast<uint32_t>(x_texels);
	block_descriptor.uniform_variables.ydim = static_cast<uint32_t>(y_texels);
	block_descriptor.uniform_variables.texel_count = static_cast<uint32_t>(x_texels * y_texels);

	block_descriptor.uniform_variables.channel_weights[0] = ERROR_WEIGHT_R;
	block_descriptor.uniform_variables.channel_weights[1] = ERROR_WEIGHT_G;
	block_descriptor.uniform_variables.channel_weights[2] = ERROR_WEIGHT_B;
	block_descriptor.uniform_variables.channel_weights[3] = ERROR_WEIGHT_A;

	for (unsigned int i = 0; i < MAX_DMI; i++) {
		decimation_mode_index[i] = -1;
	}

	unsigned int packed_bm_idx = 0;
	unsigned int packed_dm_idx = 0;

	// Trackers
	unsigned int bm_count = 0;
	unsigned int dm_count = 0;

	for (unsigned int i = 0; i < WEIGHTS_MAX_BLOCK_MODES; i++) {

		// Decode parameters
		unsigned int x_weights;
		unsigned int y_weights;
		bool is_dual_plane;
		unsigned int quant_mode;
		unsigned int weight_bits;
		bool valid = decode_block_mode_2d(i, x_weights, y_weights, is_dual_plane, quant_mode, weight_bits);

		// Always skip invalid encodings for the current block size
		if (!valid || (x_weights > x_texels) || (y_weights > y_texels))
		{
			block_descriptor.block_mode_index[i] = BLOCK_BAD_BLOCK_MODE;
			continue;
		}

		// Always skip encodings we can't physically encode based on
			// generic encoding bit availability
		if (is_dual_plane)
		{
			// This is the only check we need as only support 1 partition
			if ((109 - weight_bits) <= 0)
			{
				block_descriptor.block_mode_index[i] = BLOCK_BAD_BLOCK_MODE;
				continue;
			}
		}
		else
		{
			// This is conservative - fewer bits may be available for > 1 partition
			if ((111 - weight_bits) <= 0)
			{
				block_descriptor.block_mode_index[i] = BLOCK_BAD_BLOCK_MODE;
				continue;
			}
		}

		// Allocate and initialize the decimation table entry if we've not used it yet
		int decimation_mode = decimation_mode_index[y_weights * 16 + x_weights];
		if (decimation_mode < 0)
		{
			construct_dt_entry(x_texels, y_texels, x_weights, y_weights, block_descriptor, *wb, packed_dm_idx);
			decimation_mode_index[y_weights * 16 + x_weights] = packed_dm_idx;
			decimation_mode = packed_dm_idx;

			dm_count++;
			packed_dm_idx++;
		}

		auto& bm = block_descriptor.block_modes[packed_bm_idx];

		bm.decimation_mode = static_cast<uint32_t>(decimation_mode);
		bm.quant_mode = static_cast<uint32_t>(quant_mode);
		bm.is_dual_plane = static_cast<uint32_t>(is_dual_plane);
		bm.weight_bits = static_cast<uint32_t>(weight_bits);
		bm.mode_index = static_cast<uint32_t>(i);

		auto& dm = block_descriptor.decimation_modes[decimation_mode];

		if (is_dual_plane) {
			quant_method qm = static_cast<quant_method>(bm.quant_mode);
			dm.refprec_2planes = dm.refprec_2planes | static_cast<uint32_t>(1 << qm);
		}
		else {
			quant_method qm = static_cast<quant_method>(bm.quant_mode);
			dm.refprec_1plane = dm.refprec_1plane | static_cast<uint32_t>(1 << qm);
		}

		block_descriptor.block_mode_index[i] = static_cast<uint32_t>(packed_bm_idx);
		packed_bm_idx++;
		bm_count++;

	}

	block_descriptor.uniform_variables.block_mode_count = bm_count;
	block_descriptor.uniform_variables.decimation_mode_count = dm_count;

	// Ensure the end of the array contains valid data (should never get read)
	for (unsigned int i = dm_count; i < WEIGHTS_MAX_DECIMATION_MODES; i++)
	{
		block_descriptor.decimation_modes[i].maxprec_1plane = -1;
		block_descriptor.decimation_modes[i].maxprec_2planes = -1;
		block_descriptor.decimation_modes[i].refprec_1plane = 0;
		block_descriptor.decimation_modes[i].refprec_2planes = 0;
	}

	delete wb;
}

void construct_angular_tables(std::vector<float>& sinTable, std::vector<float>& cosTable) {
	sinTable.resize(SINCOS_STEPS * ANGULAR_STEPS);
	cosTable.resize(SINCOS_STEPS * ANGULAR_STEPS);

	for (unsigned int i = 0; i < ANGULAR_STEPS; i++) {
		float angle_step = static_cast<float>(i + 1);

		for (unsigned int j = 0; j < SINCOS_STEPS; j++) {
			// The math is identical
			float angle = (2.0f * PI / (SINCOS_STEPS - 1.0f)) * angle_step * static_cast<float>(j);

			// The indexing is now flattened
			unsigned int flat_index = j * ANGULAR_STEPS + i;

			sinTable[flat_index] = sinf(angle);
			cosTable[flat_index] = cosf(angle);
		}
	}
}