#include "averages_and_directions.h"

float dot_product(float vec1[4], float vec2[4]) {
	return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2] + vec1[3] * vec2[3];
}

float dot_product3(float vec1[4], float vec2[4]) {
	return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2];
}

std::array<float, 4> normalize(float vec[4]) {
	float norm = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2] + vec[3] * vec[3];
	if (norm != 0) {
		norm = std::sqrt(norm);
		std::array<float, 4> normvec = { vec[0] / norm, vec[1] / norm, vec[2] / norm, vec[3] / norm };
		return normvec;
	}
	else {
		std::array<float, 4> normvec = { 0, 0, 0, 0 };
		return normvec;
	}
}

static void compute_partition_averages_rgba(
	const partition_info& pi,
	const InputBlock& blk,
	int texel_count,
	float averages[BLOCK_MAX_PARTITIONS][4]
) {
	float sum[BLOCK_MAX_PARTITIONS][4]{ 0 };

	for (int i = 0; i < texel_count; i++) {
		int partition = pi.partition_of_texel[i];
		
		for (int c = 0; c < 4; c++) {
			sum[partition][c] += blk.pixels[i].data[c];
		}
	}

	for (int p = 0; p < pi.partition_count; p++) {
		int partition_texel_count = pi.partition_texel_count[p];

		for (int c = 0; c < 4; c++) {
			averages[p][c] = sum[p][c] / static_cast<float>(partition_texel_count);
		}
	}
}

void compute_avgs_and_dirs_4_comp(
	const partition_info& pi,
	const InputBlock& blk,
	int texel_count,
	partition_metrics pm[BLOCK_MAX_PARTITIONS]
) {
	size_t partition_count = pi.partition_count;

	float partition_averages[BLOCK_MAX_PARTITIONS][4];
	compute_partition_averages_rgba(pi, blk, texel_count, partition_averages);

	for (size_t partition = 0; partition < partition_count; partition++)
	{
		const uint8_t* texel_indexes = pi.texels_of_partition[partition];
		size_t texel_count_p = pi.partition_texel_count[partition];

		for (int c = 0; c < 4; c++) {
			pm[partition].avg[c] = partition_averages[partition][c];
		}

		float sum_xp[4]{ 0 };
		float sum_yp[4]{ 0 };
		float sum_zp[4]{ 0 };
		float sum_wp[4]{ 0 };

		for (size_t i = 0; i < texel_count_p; i++)
		{
			unsigned int iwt = texel_indexes[i];
			float texel_datum[4];
			for (int c = 0; c < 4; c++) {
				texel_datum[c] = blk.pixels[iwt].data[c] - partition_averages[partition][c];
			}

			if (texel_datum[0] > 0) {
				for (int c = 0; c < 4; c++) { sum_xp[c] += texel_datum[c]; }
			}
			if (texel_datum[1] > 0) {
				for (int c = 0; c < 4; c++) { sum_yp[c] += texel_datum[c]; }
			}
			if (texel_datum[2] > 0) {
				for (int c = 0; c < 4; c++) { sum_zp[c] += texel_datum[c]; }
			}
			if (texel_datum[3] > 0) {
				for (int c = 0; c < 4; c++) { sum_wp[c] += texel_datum[c]; }
			}
		}

		float prod_xp = dot_product(sum_xp, sum_xp);
		float prod_yp = dot_product(sum_yp, sum_yp);
		float prod_zp = dot_product(sum_zp, sum_zp);
		float prod_wp = dot_product(sum_wp, sum_wp);

		float best_sum = prod_xp;
		float* best_vector_ptr = sum_xp;

		if (prod_yp > best_sum) {
			best_sum = prod_yp;
			best_vector_ptr = sum_yp;
		}
		if (prod_zp > best_sum) {
			best_sum = prod_zp;
			best_vector_ptr = sum_zp;
		}
		if (prod_wp > best_sum) {
			best_sum = prod_wp;
			best_vector_ptr = sum_wp;
		}

		for (int c = 0; c < 4; c++) {
			pm[partition].dir[c] = best_vector_ptr[c];
		}
	}
}

void compute_avgs_and_dirs_3_comp_rgb(
	const partition_info& pi,
	const InputBlock& blk,
	int texel_count,
	partition_metrics pm[BLOCK_MAX_PARTITIONS]
) {
	size_t partition_count = pi.partition_count;

	float partition_averages[BLOCK_MAX_PARTITIONS][4];
	compute_partition_averages_rgba(pi, blk, texel_count, partition_averages);

	for (size_t partition = 0; partition < partition_count; partition++)
	{
		const uint8_t* texel_indexes = pi.texels_of_partition[partition];
		size_t texel_count_p = pi.partition_texel_count[partition];

		for (int c = 0; c < 4; c++) {
			pm[partition].avg[c] = partition_averages[partition][c];
		}

		float sum_xp[4]{ 0 };
		float sum_yp[4]{ 0 };
		float sum_zp[4]{ 0 };

		for (size_t i = 0; i < texel_count_p; i++)
		{
			unsigned int iwt = texel_indexes[i];
			float texel_datum[4];
			for (int c = 0; c < 4; c++) {
				texel_datum[c] = blk.pixels[iwt].data[c] - partition_averages[partition][c];
			}

			if (texel_datum[0] > 0) {
				for (int c = 0; c < 4; c++) { sum_xp[c] += texel_datum[c]; }
			}
			if (texel_datum[1] > 0) {
				for (int c = 0; c < 4; c++) { sum_yp[c] += texel_datum[c]; }
			}
			if (texel_datum[2] > 0) {
				for (int c = 0; c < 4; c++) { sum_zp[c] += texel_datum[c]; }
			}
		}

		float prod_xp = dot_product3(sum_xp, sum_xp);
		float prod_yp = dot_product3(sum_yp, sum_yp);
		float prod_zp = dot_product3(sum_zp, sum_zp);

		float best_sum = prod_xp;
		float* best_vector_ptr = sum_xp;

		if (prod_yp > best_sum) {
			best_sum = prod_yp;
			best_vector_ptr = sum_yp;
		}
		if (prod_zp > best_sum) {
			best_sum = prod_zp;
			best_vector_ptr = sum_zp;
		}

		for (int c = 0; c < 3; c++) {
			pm[partition].dir[c] = best_vector_ptr[c];
		}
		pm[partition].dir[3] = 0;
	}
}

void compute_error_squared_rgba(
	const block_descriptor& block_descriptor,
	const partition_info& pi,
	const InputBlock& blk,
	const processed_line4 uncor_plines[BLOCK_MAX_PARTITIONS],
	const processed_line4 samec_plines[BLOCK_MAX_PARTITIONS],
	float line_lengths[BLOCK_MAX_PARTITIONS],
	float& uncor_error,
	float& samec_error
) {
	size_t partition_count = pi.partition_count;
	const float* error_weights = block_descriptor.uniform_variables.channel_weights;

	float uncor_errorsum = 0;
	float samec_errorsum = 0;

	for (size_t partition = 0; partition < partition_count; partition++)
	{
		const uint8_t* texel_indexes = pi.texels_of_partition[partition];

		processed_line4 l_uncor = uncor_plines[partition];
		processed_line4 l_samec = samec_plines[partition];

		size_t texel_count = pi.partition_texel_count[partition];

		float uncor_loparam = 1e10f;
		float uncor_hiparam = -1e10f;

		for (size_t i = 0; i < texel_count; i++)
		{
			Pixel px = blk.pixels[texel_indexes[i]];

			float uncor_param = px.data[0] * l_uncor.bs[0] + px.data[1] * l_uncor.bs[1] + px.data[2] * l_uncor.bs[2] + px.data[3] * l_uncor.bs[3];

			uncor_loparam = std::min(uncor_loparam, uncor_param);
			uncor_hiparam = std::max(uncor_hiparam, uncor_param);

			float uncor_dist0 = (l_uncor.amod[0] - px.data[0]) + (uncor_param * l_uncor.bs[0]);
			float uncor_dist1 = (l_uncor.amod[1] - px.data[1]) + (uncor_param * l_uncor.bs[1]);
			float uncor_dist2 = (l_uncor.amod[2] - px.data[2]) + (uncor_param * l_uncor.bs[2]);
			float uncor_dist3 = (l_uncor.amod[3] - px.data[3]) + (uncor_param * l_uncor.bs[3]);

			float uncor_err = (error_weights[0] * uncor_dist0 * uncor_dist0)
				+ (error_weights[1] * uncor_dist1 * uncor_dist1)
				+ (error_weights[2] * uncor_dist2 * uncor_dist2)
				+ (error_weights[3] * uncor_dist3 * uncor_dist3);

			uncor_errorsum += uncor_err;

			float samec_param = px.data[0] * l_samec.bs[0] + px.data[1] * l_samec.bs[1] + px.data[2] * l_samec.bs[2] + px.data[3] * l_samec.bs[3];

			float samec_dist0 = samec_param * l_samec.bs[0] - px.data[0];
			float samec_dist1 = samec_param * l_samec.bs[1] - px.data[1];
			float samec_dist2 = samec_param * l_samec.bs[2] - px.data[2];
			float samec_dist3 = samec_param * l_samec.bs[3] - px.data[3];

			float samec_err = (error_weights[0] * samec_dist0 * samec_dist0)
				+ (error_weights[1] * samec_dist1 * samec_dist1)
				+ (error_weights[2] * samec_dist2 * samec_dist2)
				+ (error_weights[3] * samec_dist3 * samec_dist3);

			samec_errorsum += samec_err;
		}

		float uncor_linelen = uncor_hiparam - uncor_loparam;
		line_lengths[partition] = std::max(uncor_linelen, 1e-7f);
	}

	uncor_error = uncor_errorsum;
	samec_error = samec_errorsum;
}

void compute_error_squared_rgb(
	const block_descriptor& block_descriptor,
	const partition_info& pi,
	const InputBlock& blk,
	partition_lines3 plines[BLOCK_MAX_PARTITIONS],
	float& uncor_error,
	float& samec_error
) {
	size_t partition_count = pi.partition_count;
	const float* error_weights = block_descriptor.uniform_variables.channel_weights;

	float uncor_errorsum = 0;
	float samec_errorsum = 0;

	for (size_t partition = 0; partition < partition_count; partition++)
	{
		partition_lines3& pl = plines[partition];
		const uint8_t* texel_indexes = pi.texels_of_partition[partition];
		size_t texel_count = pi.partition_texel_count[partition];

		processed_line3 l_uncor = pl.uncor_pline;
		processed_line3 l_samec = pl.samec_pline;

		float uncor_loparam = 1e10f;
		float uncor_hiparam = -1e10f;

		for (size_t i = 0; i < texel_count; i++)
		{
			Pixel px = blk.pixels[texel_indexes[i]];

			float uncor_param = px.data[0] * l_uncor.bs[0] + px.data[1] * l_uncor.bs[1] + px.data[2] * l_uncor.bs[2];

			uncor_loparam = std::min(uncor_loparam, uncor_param);
			uncor_hiparam = std::max(uncor_hiparam, uncor_param);

			float uncor_dist0 = (l_uncor.amod[0] - px.data[0]) + (uncor_param * l_uncor.bs[0]);
			float uncor_dist1 = (l_uncor.amod[1] - px.data[1]) + (uncor_param * l_uncor.bs[1]);
			float uncor_dist2 = (l_uncor.amod[2] - px.data[2]) + (uncor_param * l_uncor.bs[2]);

			float uncor_err = (error_weights[0] * uncor_dist0 * uncor_dist0)
				+ (error_weights[1] * uncor_dist1 * uncor_dist1)
				+ (error_weights[2] * uncor_dist2 * uncor_dist2);

			uncor_errorsum += uncor_err;


			float samec_param = px.data[0] * l_samec.bs[0] + px.data[1] * l_samec.bs[1] + px.data[2] * l_samec.bs[2];

			float samec_dist0 = samec_param * l_samec.bs[0] - px.data[0];
			float samec_dist1 = samec_param * l_samec.bs[1] - px.data[1];
			float samec_dist2 = samec_param * l_samec.bs[2] - px.data[2];

			float samec_err = (error_weights[0] * samec_dist0 * samec_dist0)
				+ (error_weights[1] * samec_dist1 * samec_dist1)
				+ (error_weights[2] * samec_dist2 * samec_dist2);

			samec_errorsum += samec_err;
		}

		float uncor_linelen = uncor_hiparam - uncor_loparam;
		pl.line_length = std::max(uncor_linelen, 1e-7f);
	}

	uncor_error = uncor_errorsum;
	samec_error = samec_errorsum;
}