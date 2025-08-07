#include "astc.h"

struct partition_metrics
{
	float avg[4];
	float dir[4];
};

struct line3
{
	float a[4];
	float b[4];
};

struct line4
{
	float a[4];
	float b[4];
};

struct processed_line3
{
	float amod[4];
	float bs[4];
};

struct processed_line4
{
	float amod[4];
	float bs[4];
};

struct partition_lines3
{
	line3 uncor_line;
	line3 samec_line;
	processed_line3 uncor_pline;
	processed_line3 samec_pline;

	float line_length;
};

float dot_product(float vec1[4], float vec2[4]);

float dot_product3(float vec1[4], float vec2[4]);

std::array<float, 4> normalize(float vec[4]);

void compute_avgs_and_dirs_3_comp_rgb(
	const partition_info& pi,
	const InputBlock& blk,
	int texel_count,
	partition_metrics pm[BLOCK_MAX_PARTITIONS]
);

void compute_avgs_and_dirs_4_comp(
	const partition_info& pi,
	const InputBlock& blk,
	int texel_count,
	partition_metrics pm[BLOCK_MAX_PARTITIONS]
);

void compute_error_squared_rgba(
	const block_descriptor& block_descriptor,
	const partition_info& pi,
	const InputBlock& blk,
	const processed_line4 uncor_plines[BLOCK_MAX_PARTITIONS],
	const processed_line4 samec_plines[BLOCK_MAX_PARTITIONS],
	float line_lengths[BLOCK_MAX_PARTITIONS],
	float& uncor_error,
	float& samec_error
);

void compute_error_squared_rgb(
	const block_descriptor& block_descriptor,
	const partition_info& pi,
	const InputBlock& blk,
	partition_lines3 plines[BLOCK_MAX_PARTITIONS],
	float& uncor_error,
	float& samec_error
);