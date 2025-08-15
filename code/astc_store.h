#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include <memory>

int store_image(
	unsigned int block_x,
	unsigned int block_y,
	unsigned int dim_x,
	unsigned int dim_y,
	uint8_t* data,
	size_t data_len,
	const std::string& filename
);

#if defined(EMSCRIPTEN)
struct AstcFile {
	std::unique_ptr<uint8_t[]> data;
	size_t size;
};

AstcFile create_astc_file_in_memory(
	unsigned int block_x,
	unsigned int block_y,
	unsigned int dim_x,
	unsigned int dim_y,
	uint8_t* data,
	size_t data_len
);
#endif