#include "astc.h"
#include "averages_and_directions.h"

#include <iostream>

// Calculates the weighted squared difference between two colors
static float dot_product_squared_difference(
	const float color1[4],
	const float color2[4],
	const float channel_weights[4]
) {
	float diff_r = color1[0] - color2[0];
	float diff_g = color1[1] - color2[1];
	float diff_b = color1[2] - color2[2];
	float diff_a = color1[3] - color2[3];

	return (diff_r * diff_r * channel_weights[0]) +
		(diff_g * diff_g * channel_weights[1]) +
		(diff_b * diff_b * channel_weights[2]) +
		(diff_a * diff_a * channel_weights[3]);
}

/**
 * @brief Pick some initial kmeans cluster centers.
 */
void kmeans_init(
	const InputBlock& blk,
	unsigned int texel_count,
	unsigned int partition_count,
	const float channel_weights[4],
	std::vector<std::array<float, 4>>& cluster_centers
) {
	
	cluster_centers.clear();
	cluster_centers.reserve(partition_count);
	
	std::vector<float> distances(texel_count);

	// Pick a random sample as first cluster center; 145897 from random.org
	unsigned int sample_index = 145897 % texel_count;

	std::array<float, 4> center_color;
	for (int i = 0; i < 4; ++i) {
		center_color[i] = blk.pixels[sample_index][i];
	}
	cluster_centers.push_back(center_color);

	// Compute the distance to the first cluster center
	float distance_sum = 0.0f;
	for (unsigned int i = 0; i < texel_count; ++i) {
		float distance = dot_product_squared_difference(blk.pixels[i], center_color.data(), channel_weights);
		distance_sum += distance;
		distances[i] = distance;
	}

	// More numbers from random.org for weighted-random center selection
	const float cluster_cutoffs[9]{
		0.626220f, 0.932770f, 0.275454f,
		0.318558f, 0.240113f, 0.009190f,
		0.347661f, 0.731960f, 0.156391f
	};

	unsigned int cutoff_index = (cluster_centers.size() - 1) + 3 * (partition_count - 2);

	// Pick the remaining samples as needed
	while (true) {
		// Pick the next center in a weighted-random fashion.
		float summa = 0.0f;
		float distance_cutoff = distance_sum * cluster_cutoffs[cutoff_index++];

		unsigned int next_sample_index = 0;
		for (; next_sample_index < texel_count; ++next_sample_index) {
			summa += distances[next_sample_index];
			if (summa >= distance_cutoff) {
				break;
			}
		}

		// Clamp to a valid range and store the selected cluster center
		next_sample_index = std::min(next_sample_index, texel_count - 1);

		for (int i = 0; i < 4; ++i) {
			center_color[i] = blk.pixels[next_sample_index][i];
		}
		cluster_centers.push_back(center_color);

		if (cluster_centers.size() >= partition_count) {
			break;
		}

		// Compute the distance to the new cluster center, keeping the min distance
		distance_sum = 0.0f;
		for (unsigned int i = 0; i < texel_count; ++i) {
			float distance = dot_product_squared_difference(blk.pixels[i], center_color.data(), channel_weights);
			distances[i] = std::min(distance, distances[i]);
			distance_sum += distances[i];
		}
	}
}

void kmeans_assign(
	const InputBlock& blk,
	unsigned int texel_count,
	unsigned int partition_count,
	const float channel_weights[4],
	const std::vector<std::array<float, 4>>& cluster_centers,
	uint8_t partition_of_texel[BLOCK_MAX_TEXELS]
) {
	uint8_t partition_texel_count[BLOCK_MAX_PARTITIONS]{ 0 };

	// Find the best partition for every texel
	for (unsigned int i = 0; i < texel_count; ++i) {
		float best_distance = std::numeric_limits<float>::max();
		unsigned int best_partition = 0;

		// For each texel, check its distance to every cluster center
		for (unsigned int j = 0; j < partition_count; ++j) {
			float distance = dot_product_squared_difference(
				blk.pixels[i],
				cluster_centers[j].data(),
				channel_weights
			);

			if (distance < best_distance) {
				best_distance = distance;
				best_partition = j;
			}
		}

		// Assign the texel to the partition with the closest center
		partition_of_texel[i] = static_cast<uint8_t>(best_partition);
		partition_texel_count[best_partition]++;
	}

	// This is the crucial safety check from the original code.
	// It finds empty partitions and "steals" a texel to ensure every partition has at least one member.
	bool problem_case_found;
	do {
		problem_case_found = false;
		for (unsigned int i = 0; i < partition_count; ++i) {
			if (partition_texel_count[i] == 0) {
				problem_case_found = true;

				uint8_t old_partition_of_texel_i = partition_of_texel[i];
				partition_texel_count[old_partition_of_texel_i]--;
				partition_texel_count[i]++;
				partition_of_texel[i] = static_cast<uint8_t>(i);
			}
		}
	} while (problem_case_found);
}

static void kmeans_update(
	const InputBlock& blk,
	unsigned int texel_count,
	unsigned int partition_count,
	std::vector<std::array<float, 4>>& cluster_centers,
	const uint8_t partition_of_texel[BLOCK_MAX_TEXELS]
) {

	std::vector<std::array<float, 4>> color_sum(partition_count, { 0.0f, 0.0f, 0.0f, 0.0f });
	uint8_t partition_texel_count[BLOCK_MAX_PARTITIONS]{ 0 };

	// Find the center of gravity (sum of colors) in each cluster.
	for (unsigned int i = 0; i < texel_count; ++i) {
		uint8_t partition = partition_of_texel[i];

		// Add the texel's color to the appropriate sum, component by component.
		for (int c = 0; c < 4; ++c) {
			color_sum[partition][c] += blk.pixels[i][c];
		}

		partition_texel_count[partition]++;
	}

	// Set the center of gravity (the average color) to be the new cluster center.
	for (unsigned int i = 0; i < partition_count; ++i) {

		//safety check
		unsigned int count = partition_texel_count[i];
		if (count == 0) {
			cluster_centers[i] = { 0.0f, 0.0f, 0.0f, 0.0f };
			continue;
		}

		float scale = 1.0f / static_cast<float>(count);

		// Calculate the average by scaling the sum.
		for (int c = 0; c < 4; ++c) {
			cluster_centers[i][c] = color_sum[i][c] * scale;
		}
	}

}

/**
 * @brief Population bit count.
 *
 * @param v   The value to population count.
 *
 * @return The number of 1 bits.
 */
static inline int popcount(uint64_t v)
{
	uint64_t mask1 = 0x5555555555555555ULL;
	uint64_t mask2 = 0x3333333333333333ULL;
	uint64_t mask3 = 0x0F0F0F0F0F0F0F0FULL;
	v -= (v >> 1) & mask1;
	v = (v & mask2) + ((v >> 2) & mask2);
	v += v >> 4;
	v &= mask3;
	v *= 0x0101010101010101ULL;
	v >>= 56;
	return static_cast<int>(v);
}

/**
 * @brief Compute bit-mismatch for partitioning in 2-partition mode.
 *
 * @param a   The texel assignment bitvector for the block.
 * @param b   The texel assignment bitvector for the partition table.
 *
 * @return    The number of bit mismatches.
 */
static inline uint8_t partition_mismatch2(
	const uint64_t a[2],
	const uint64_t b[2]
) {
	int v1 = popcount(a[0] ^ b[0]) + popcount(a[1] ^ b[1]);
	int v2 = popcount(a[0] ^ b[1]) + popcount(a[1] ^ b[0]);

	// Divide by 2 because XOR always counts errors twice, once when missing
	// in the expected position, and again when present in the wrong partition
	return static_cast<uint8_t>(std::min(v1, v2) / 2);
}

/**
 * @brief Compute bit-mismatch for partitioning in 3-partition mode.
 *
 * @param a   The texel assignment bitvector for the block.
 * @param b   The texel assignment bitvector for the partition table.
 *
 * @return    The number of bit mismatches.
 */
static inline uint8_t partition_mismatch3(
	const uint64_t a[3],
	const uint64_t b[3]
) {
	int p00 = popcount(a[0] ^ b[0]);
	int p01 = popcount(a[0] ^ b[1]);
	int p02 = popcount(a[0] ^ b[2]);

	int p10 = popcount(a[1] ^ b[0]);
	int p11 = popcount(a[1] ^ b[1]);
	int p12 = popcount(a[1] ^ b[2]);

	int p20 = popcount(a[2] ^ b[0]);
	int p21 = popcount(a[2] ^ b[1]);
	int p22 = popcount(a[2] ^ b[2]);

	int s0 = p11 + p22;
	int s1 = p12 + p21;
	int v0 = std::min(s0, s1) + p00;

	int s2 = p10 + p22;
	int s3 = p12 + p20;
	int v1 = std::min(s2, s3) + p01;

	int s4 = p10 + p21;
	int s5 = p11 + p20;
	int v2 = std::min(s4, s5) + p02;

	// Divide by 2 because XOR always counts errors twice, once when missing
	// in the expected position, and again when present in the wrong partition
	return static_cast<uint8_t>(std::min(v0, std::min(v1, v2)) / 2);
}

/**
 * @brief Compute bit-mismatch for partitioning in 4-partition mode.
 *
 * @param a   The texel assignment bitvector for the block.
 * @param b   The texel assignment bitvector for the partition table.
 *
 * @return    The number of bit mismatches.
 */
static inline uint8_t partition_mismatch4(
	const uint64_t a[4],
	const uint64_t b[4]
) {
	int p00 = popcount(a[0] ^ b[0]);
	int p01 = popcount(a[0] ^ b[1]);
	int p02 = popcount(a[0] ^ b[2]);
	int p03 = popcount(a[0] ^ b[3]);

	int p10 = popcount(a[1] ^ b[0]);
	int p11 = popcount(a[1] ^ b[1]);
	int p12 = popcount(a[1] ^ b[2]);
	int p13 = popcount(a[1] ^ b[3]);

	int p20 = popcount(a[2] ^ b[0]);
	int p21 = popcount(a[2] ^ b[1]);
	int p22 = popcount(a[2] ^ b[2]);
	int p23 = popcount(a[2] ^ b[3]);

	int p30 = popcount(a[3] ^ b[0]);
	int p31 = popcount(a[3] ^ b[1]);
	int p32 = popcount(a[3] ^ b[2]);
	int p33 = popcount(a[3] ^ b[3]);

	int mx23 = std::min(p22 + p33, p23 + p32);
	int mx13 = std::min(p21 + p33, p23 + p31);
	int mx12 = std::min(p21 + p32, p22 + p31);
	int mx03 = std::min(p20 + p33, p23 + p30);
	int mx02 = std::min(p20 + p32, p22 + p30);
	int mx01 = std::min(p21 + p30, p20 + p31);

	int v0 = p00 + std::min(p11 + mx23, std::min(p12 + mx13, p13 + mx12));
	int v1 = p01 + std::min(p10 + mx23, std::min(p12 + mx03, p13 + mx02));
	int v2 = p02 + std::min(p11 + mx03, std::min(p10 + mx13, p13 + mx01));
	int v3 = p03 + std::min(p11 + mx02, std::min(p12 + mx01, p10 + mx12));

	// Divide by 2 because XOR always counts errors twice, once when missing
	// in the expected position, and again when present in the wrong partition
	return static_cast<uint8_t>(std::min(v0, std::min(v1, std::min(v2, v3))) / 2);
}

static void count_partition_mismatch_bits(
	const block_descriptor& block_descriptor,
	unsigned int partition_count,
	const uint64_t bitmaps[BLOCK_MAX_PARTITIONS],
	uint8_t mismatch_counts[BLOCK_MAX_PARTITIONINGS]
) {
	unsigned int active_count = block_descriptor.partitioning_count_selected[partition_count - 1];

	if (partition_count == 2)
	{
		for (unsigned int i = 0; i < active_count; i++)
		{
			mismatch_counts[i] = partition_mismatch2(bitmaps, block_descriptor.coverage_bitmaps_2[i]);
			assert(mismatch_counts[i] < BLOCK_MAX_KMEANS_TEXELS);
			assert(mismatch_counts[i] < block_descriptor.uniform_variables.texel_count);
		}
	}
	else if (partition_count == 3)
	{
		for (unsigned int i = 0; i < active_count; i++)
		{
			mismatch_counts[i] = partition_mismatch3(bitmaps, block_descriptor.coverage_bitmaps_3[i]);
			assert(mismatch_counts[i] < BLOCK_MAX_KMEANS_TEXELS);
			assert(mismatch_counts[i] < block_descriptor.uniform_variables.texel_count);
		}
	}
	else
	{
		for (unsigned int i = 0; i < active_count; i++)
		{
			mismatch_counts[i] = partition_mismatch4(bitmaps, block_descriptor.coverage_bitmaps_4[i]);
			assert(mismatch_counts[i] < BLOCK_MAX_KMEANS_TEXELS);
			assert(mismatch_counts[i] < block_descriptor.uniform_variables.texel_count);
		}
	}
}

static unsigned int get_partition_ordering_by_mismatch_bits(
	unsigned int texel_count,
	unsigned int partitioning_count,
	const uint8_t mismatch_count[BLOCK_MAX_PARTITIONINGS],
	uint16_t partition_ordering[BLOCK_MAX_PARTITIONINGS]
) {
	uint16_t mscount[BLOCK_MAX_KMEANS_TEXELS]{ 0 };

	// Create the histogram of mismatch counts
	for (unsigned int i = 0; i < partitioning_count; i++)
	{
		mscount[mismatch_count[i]]++;
	}

	// Create a running sum from the histogram array
	// Indices store previous values only; i.e. exclude self after sum
	uint16_t sum = 0;
	for (unsigned int i = 0; i < texel_count; i++)
	{
		uint16_t cnt = mscount[i];
		mscount[i] = sum;
		sum += cnt;
	}

	// Use the running sum as the index, incrementing after read to allow
	// sequential entries with the same count
	for (unsigned int i = 0; i < partitioning_count; i++)
	{
		unsigned int idx = mscount[mismatch_count[i]]++;
		partition_ordering[idx] = static_cast<uint16_t>(i);
	}

	return partitioning_count;
}

/**
 * @brief Use k-means clustering to compute a partition ordering for a block..
 *
 * @param      bsd                  The block size information.
 * @param      blk                  The image block color data to compress.
 * @param      partition_count      The desired number of partitions in the block.
 * @param[out] partition_ordering   The list of recommended partition indices, in priority order.
 *
 * @return The number of active partitionings in this selection.
 */
static unsigned int compute_kmeans_partition_ordering(
	const block_descriptor& block_descriptor,
	const InputBlock& blk,
	unsigned int partition_count,
	uint16_t partition_ordering[BLOCK_MAX_PARTITIONINGS]
) {
	std::vector<std::array<float, 4>> cluster_centers = {};
	uint8_t texel_partitions[BLOCK_MAX_TEXELS];

	// Use three passes of k-means clustering to partition the block data
	for (unsigned int i = 0; i < 3; i++)
	{
		if (i == 0)
		{
			kmeans_init(blk, block_descriptor.uniform_variables.texel_count, partition_count, block_descriptor.uniform_variables.channel_weights, cluster_centers);
		}
		else
		{
			kmeans_update(blk, block_descriptor.uniform_variables.texel_count, partition_count, cluster_centers, texel_partitions);
		}

		kmeans_assign(blk, block_descriptor.uniform_variables.texel_count, partition_count, block_descriptor.uniform_variables.channel_weights, cluster_centers, texel_partitions);
	}

	// Construct the block bitmaps of texel assignments to each partition
	uint64_t bitmaps[BLOCK_MAX_PARTITIONS]{ 0 };
	unsigned int texels_to_process = std::min(block_descriptor.uniform_variables.texel_count, BLOCK_MAX_KMEANS_TEXELS);
	for (unsigned int i = 0; i < texels_to_process; i++)
	{
		unsigned int idx = block_descriptor.kmeans_texels[i];
		bitmaps[texel_partitions[idx]] |= 1ULL << i;
	}

	// Count the mismatch between the block and the format's partition tables
	uint8_t mismatch_counts[BLOCK_MAX_PARTITIONINGS];
	count_partition_mismatch_bits(block_descriptor, partition_count, bitmaps, mismatch_counts);

	// Sort the partitions based on the number of mismatched bits
	return get_partition_ordering_by_mismatch_bits(
		texels_to_process,
		block_descriptor.partitioning_count_selected[partition_count - 1],
		mismatch_counts, partition_ordering);
}

/**
 * @brief Insert a partitioning into an order list of results, sorted by error.
 *
 * @param      max_values      The max number of entries in the best result arrays.
 * @param      this_error      The error of the new entry.
 * @param      this_partition  The partition ID of the new entry.
 * @param[out] best_errors     The array of best error values.
 * @param[out] best_partitions The array of best partition values.
 */
static void insert_result(
	unsigned int max_values,
	float this_error,
	unsigned int this_partition,
	float* best_errors,
	unsigned int* best_partitions)
{
	// Don't bother searching if the current worst error beats the new error
	if (this_error >= best_errors[max_values - 1])
	{
		return;
	}

	// Else insert into the list in error-order
	for (unsigned int i = 0; i < max_values; i++)
	{
		// Existing result is better - move on ...
		if (this_error > best_errors[i])
		{
			continue;
		}

		// Move existing results down one
		for (unsigned int j = max_values - 1; j > i; j--)
		{
			best_errors[j] = best_errors[j - 1];
			best_partitions[j] = best_partitions[j - 1];
		}

		// Insert new result
		best_errors[i] = this_error;
		best_partitions[i] = this_partition;
		break;
	}
}

unsigned int find_best_partition_candidates(
	const block_descriptor& block_descriptor,
	const InputBlock& blk,
	unsigned int partition_count,
	unsigned int partition_search_limit,
	unsigned int best_partitions[TUNE_MAX_PARTITIONING_CANDIDATES],
	unsigned int requested_candidates
) {
	// Constant used to estimate quantization error for a given partitioning; the optimal value for
	// this depends on bitrate. These values have been determined empirically.
	unsigned int texels_per_block = block_descriptor.uniform_variables.texel_count;
	float weight_imprecision_estim = 0.055f;
	if (texels_per_block <= 20)
	{
		weight_imprecision_estim = 0.03f;
	}
	else if (texels_per_block <= 31)
	{
		weight_imprecision_estim = 0.04f;
	}
	else if (texels_per_block <= 41)
	{
		weight_imprecision_estim = 0.05f;
	}

	weight_imprecision_estim = weight_imprecision_estim * weight_imprecision_estim;

	uint16_t partition_sequence[BLOCK_MAX_PARTITIONINGS];
	unsigned int sequence_len = compute_kmeans_partition_ordering(block_descriptor, blk, partition_count, partition_sequence);
	partition_search_limit = std::min(partition_search_limit, sequence_len);
	requested_candidates = std::min(partition_search_limit, requested_candidates);

	bool uses_alpha = blk.constant_alpha == 0;

	// Partitioning errors assuming uncorrelated-chrominance endpoints
	float uncor_best_errors[TUNE_MAX_PARTITIONING_CANDIDATES];
	unsigned int uncor_best_partitions[TUNE_MAX_PARTITIONING_CANDIDATES];

	// Partitioning errors assuming same-chrominance endpoints
	float samec_best_errors[TUNE_MAX_PARTITIONING_CANDIDATES];
	unsigned int samec_best_partitions[TUNE_MAX_PARTITIONING_CANDIDATES];

	for (unsigned int i = 0; i < requested_candidates; i++)
	{
		uncor_best_errors[i] = ERROR_CALC_DEFAULT;
		samec_best_errors[i] = ERROR_CALC_DEFAULT;
	}

	if (uses_alpha)
	{
		for (unsigned int i = 0; i < partition_search_limit; i++)
		{
			unsigned int partition = partition_sequence[i];
			const auto& pi = block_descriptor.get_raw_partition_info(partition_count, partition);

			partition_metrics pms[BLOCK_MAX_PARTITIONS];
			compute_avgs_and_dirs_4_comp(pi, blk, block_descriptor.uniform_variables.texel_count, pms);

			line4 uncor_lines[BLOCK_MAX_PARTITIONS];
			line4 samec_lines[BLOCK_MAX_PARTITIONS];

			processed_line4 uncor_plines[BLOCK_MAX_PARTITIONS];
			processed_line4 samec_plines[BLOCK_MAX_PARTITIONS];

			float line_lengths[BLOCK_MAX_PARTITIONS];

			for (unsigned int j = 0; j < partition_count; j++)
			{
				partition_metrics& pm = pms[j];

				std::array<float, 4> norm_avg = normalize(pm.avg);
				std::array<float, 4> norm_dir = normalize(pm.dir);

				for (int c = 0; c < 4; c++) {
					uncor_lines[j].a[c] = pm.avg[c];
					uncor_lines[j].b[c] = norm_dir[c];

					samec_lines[j].a[c] = 0;
					samec_lines[j].b[c] = norm_avg[c];
				}

				float uncor_dot = dot_product(uncor_lines[j].a, uncor_lines[j].b);

				for (int c = 0; c < 4; c++) {
					uncor_plines[j].amod[c] = uncor_lines[j].a[c] - uncor_lines[j].b[c] * uncor_dot;
					uncor_plines[j].bs[c] = uncor_lines[j].b[c];

					samec_plines[j].amod[c] = 0;
					samec_plines[j].bs[c] = samec_lines[j].b[c];
				}
			}

			float uncor_error = 0.0f;
			float samec_error = 0.0f;

			compute_error_squared_rgba(block_descriptor, pi, blk, uncor_plines, samec_plines, line_lengths, uncor_error, samec_error);

			// Compute an estimate of error introduced by weight quantization imprecision.
			// This error is computed as follows, for each partition
			//     1: compute the principal-axis vector (full length) in error-space
			//     2: convert the principal-axis vector to regular RGB-space
			//     3: scale the vector by a constant that estimates average quantization error
			//     4: for each texel, square the vector, then do a dot-product with the texel's
			//        error weight; sum up the results across all texels.
			//     4(optimized): square the vector once, then do a dot-product with the average
			//        texel error, then multiply by the number of texels.

			for (unsigned int j = 0; j < partition_count; j++)
			{
				float tpp = static_cast<float>(pi.partition_texel_count[j]);
				float error_weight = tpp * weight_imprecision_estim;

				float uncor_vector[4];
				float samec_vector[4];

				for (int c = 0; c < 4; c++) {
					uncor_vector[c] = uncor_lines[j].b[c] * line_lengths[j];
					samec_vector[c] = samec_lines[j].b[c] * line_lengths[j];
				}

				uncor_error += dot_product(uncor_vector, uncor_vector) * error_weight;
				samec_error += dot_product(samec_vector, samec_vector) * error_weight;
			}

			insert_result(requested_candidates, uncor_error, partition, uncor_best_errors, uncor_best_partitions);
			insert_result(requested_candidates, samec_error, partition, samec_best_errors, samec_best_partitions);
		}
	}
	else
	{
		for (unsigned int i = 0; i < partition_search_limit; i++)
		{
			unsigned int partition = partition_sequence[i];
			const auto& pi = block_descriptor.get_raw_partition_info(partition_count, partition);

			partition_metrics pms[BLOCK_MAX_PARTITIONS];
			compute_avgs_and_dirs_3_comp_rgb(pi, blk, block_descriptor.uniform_variables.texel_count, pms);

			partition_lines3 plines[BLOCK_MAX_PARTITIONS];

			for (unsigned int j = 0; j < partition_count; j++)
			{
				partition_metrics& pm = pms[j];
				partition_lines3& pl = plines[j];

				std::array<float, 4> norm_avg = normalize(pm.avg);
				std::array<float, 4> norm_dir = normalize(pm.dir);

				for (int c = 0; c < 4; c++) {
					pl.uncor_line.a[c] = pm.avg[c];
					pl.uncor_line.b[c] = norm_dir[c];

					pl.samec_line.a[c] = 0;
					pl.samec_line.b[c] = norm_avg[c];
				}

				float uncor_dot = dot_product3(pl.uncor_line.a, pl.uncor_line.b);

				for (int c = 0; c < 4; c++) {
					pl.uncor_pline.amod[c] = pl.uncor_line.a[c] - pl.uncor_line.b[c] * uncor_dot;
					pl.uncor_pline.bs[c] = pl.uncor_line.b[c];

					pl.samec_pline.amod[c] = 0;
					pl.samec_pline.bs[c] = pl.samec_line.b[c];
				}
			}

			float uncor_error = 0.0f;
			float samec_error = 0.0f;

			compute_error_squared_rgb(block_descriptor, pi, blk, plines, uncor_error, samec_error);

			// Compute an estimate of error introduced by weight quantization imprecision.
			// This error is computed as follows, for each partition
			//     1: compute the principal-axis vector (full length) in error-space
			//     2: convert the principal-axis vector to regular RGB-space
			//     3: scale the vector by a constant that estimates average quantization error
			//     4: for each texel, square the vector, then do a dot-product with the texel's
			//        error weight; sum up the results across all texels.
			//     4(optimized): square the vector once, then do a dot-product with the average
			//        texel error, then multiply by the number of texels.

			for (unsigned int j = 0; j < partition_count; j++)
			{
				partition_lines3& pl = plines[j];
				float tpp = static_cast<float>(pi.partition_texel_count[j]);
				float error_weight = tpp * weight_imprecision_estim;

				float uncor_vector[4];
				float samec_vector[4];

				for (int c = 0; c < 4; c++) {
					uncor_vector[c] = pl.uncor_line.b[c] * pl.line_length;
					samec_vector[c] = pl.samec_line.b[c] * pl.line_length;
				}

				uncor_error += dot_product3(uncor_vector, uncor_vector) * error_weight;
				samec_error += dot_product3(samec_vector, samec_vector) * error_weight;
			}

			insert_result(requested_candidates, uncor_error, partition, uncor_best_errors, uncor_best_partitions);
			insert_result(requested_candidates, samec_error, partition, samec_best_errors, samec_best_partitions);
		}
	}

	unsigned int interleave[2 * TUNE_MAX_PARTITIONING_CANDIDATES];
	for (unsigned int i = 0; i < requested_candidates; i++)
	{
		interleave[2 * i] = block_descriptor.get_raw_partition_info(partition_count, uncor_best_partitions[i]).partition_index;
		interleave[2 * i + 1] = block_descriptor.get_raw_partition_info(partition_count, samec_best_partitions[i]).partition_index;
	}

	uint64_t bitmasks[1024 / 64]{ 0 };
	unsigned int emitted = 0;

	// Deduplicate the first "requested" entries
	for (unsigned int i = 0; i < requested_candidates * 2; i++)
	{
		unsigned int partition = interleave[i];

		unsigned int word = partition / 64;
		unsigned int bit = partition % 64;

		bool written = bitmasks[word] & (1ull << bit);

		if (!written)
		{
			best_partitions[emitted] = partition;
			bitmasks[word] |= 1ull << bit;
			emitted++;

			if (emitted == requested_candidates)
			{
				break;
			}
		}
	}

	return emitted;
}