#include "astc_store.h"
#include <stdexcept>


struct astc_header
{
	uint8_t magic[4];
	uint8_t block_x;
	uint8_t block_y;
	uint8_t block_z;
	uint8_t dim_x[3];			// dims = dim[0] + (dim[1] << 8) + (dim[2] << 16)
	uint8_t dim_y[3];			// Sizes are given in texels;
	uint8_t dim_z[3];			// block count is inferred
};

static const uint32_t ASTC_MAGIC_ID = 0x5CA1AB13;

int store_image(
	unsigned int block_x,
	unsigned int block_y,
	unsigned int dim_x,
	unsigned int dim_y,
	uint8_t* data, 
	size_t data_len,
	const std::string& filename
) {
	unsigned int block_z = 1;
	unsigned int dim_z = 1;

	astc_header hdr;
	hdr.magic[0] = ASTC_MAGIC_ID & 0xFF;
	hdr.magic[1] = (ASTC_MAGIC_ID >> 8) & 0xFF;
	hdr.magic[2] = (ASTC_MAGIC_ID >> 16) & 0xFF;
	hdr.magic[3] = (ASTC_MAGIC_ID >> 24) & 0xFF;

	hdr.block_x = static_cast<uint8_t>(block_x);
	hdr.block_y = static_cast<uint8_t>(block_y);
	hdr.block_z = static_cast<uint8_t>(block_z);

	hdr.dim_x[0] = dim_x & 0xFF;
	hdr.dim_x[1] = (dim_x >> 8) & 0xFF;
	hdr.dim_x[2] = (dim_x >> 16) & 0xFF;

	hdr.dim_y[0] = dim_y & 0xFF;
	hdr.dim_y[1] = (dim_y >> 8) & 0xFF;
	hdr.dim_y[2] = (dim_y >> 16) & 0xFF;

	hdr.dim_z[0] = dim_z & 0xFF;
	hdr.dim_z[1] = (dim_z >> 8) & 0xFF;
	hdr.dim_z[2] = (dim_z >> 16) & 0xFF;

	std::ofstream file(filename.c_str(), std::ios::out | std::ios::binary);
	if (!file)
	{
		throw std::runtime_error("File open failed: " + filename);
		return 1;
	}

	file.write(reinterpret_cast<char*>(&hdr), sizeof(astc_header));
	file.write(reinterpret_cast<char*>(data), data_len);
	return 0;
}

#if defined(EMSCRIPTEN)
AstcFile create_astc_file_in_memory(
	unsigned int block_x,
	unsigned int block_y,
	unsigned int dim_x,
	unsigned int dim_y,
	uint8_t* data,
	size_t data_len
) {
	const size_t headerSize = 16;
	const size_t finalFileSize = headerSize + data_len;
	auto finalFileBuffer = std::make_unique<uint8_t[]>(finalFileSize);
	uint8_t* ptr = finalFileBuffer.get();

	*ptr++ = ASTC_MAGIC_ID & 0xFF;
	*ptr++ = (ASTC_MAGIC_ID >> 8) & 0xFF;
	*ptr++ = (ASTC_MAGIC_ID >> 16) & 0xFF;
	*ptr++ = (ASTC_MAGIC_ID >> 24) & 0xFF;

	*ptr++ = static_cast<uint8_t>(block_x);
	*ptr++ = static_cast<uint8_t>(block_y);
	*ptr++ = 1; // block_z

	*ptr++ = dim_x & 0xFF;
	*ptr++ = (dim_x >> 8) & 0xFF;
	*ptr++ = (dim_x >> 16) & 0xFF;

	*ptr++ = dim_y & 0xFF;
	*ptr++ = (dim_y >> 8) & 0xFF;
	*ptr++ = (dim_y >> 16) & 0xFF;

	uint32_t dim_z = 1;
	*ptr++ = dim_z & 0xFF;
	*ptr++ = (dim_z >> 8) & 0xFF;
	*ptr++ = (dim_z >> 16) & 0xFF;

	// Copy the raw compressed data after the correctly formatted header.
	memcpy(ptr, data, data_len);

	return AstcFile{ std::move(finalFileBuffer), finalFileSize };
}
#endif