#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <cstring>

int store_image(
	unsigned int block_x,
	unsigned int block_y,
	unsigned int dim_x,
	unsigned int dim_y,
	uint8_t* data,
	size_t data_len,
	const std::string& filename
);