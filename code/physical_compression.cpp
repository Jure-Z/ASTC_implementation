#include "astc.h"


struct btq_count
{
	/** @brief The number of bits. */
	uint8_t bits : 6;

	/** @brief The number of trits. */
	uint8_t trits : 1;

	/** @brief The number of quints. */
	uint8_t quints : 1;
};

/**
 * @brief The table of bits, trits, and quints needed for a quant encode.
 */
static const std::array<btq_count, 21> btq_counts{ {
	{ 1, 0, 0 }, // QUANT_2
	{ 0, 1, 0 }, // QUANT_3
	{ 2, 0, 0 }, // QUANT_4
	{ 0, 0, 1 }, // QUANT_5
	{ 1, 1, 0 }, // QUANT_6
	{ 3, 0, 0 }, // QUANT_8
	{ 1, 0, 1 }, // QUANT_10
	{ 2, 1, 0 }, // QUANT_12
	{ 4, 0, 0 }, // QUANT_16
	{ 2, 0, 1 }, // QUANT_20
	{ 3, 1, 0 }, // QUANT_24
	{ 5, 0, 0 }, // QUANT_32
	{ 3, 0, 1 }, // QUANT_40
	{ 4, 1, 0 }, // QUANT_48
	{ 6, 0, 0 }, // QUANT_64
	{ 4, 0, 1 }, // QUANT_80
	{ 5, 1, 0 }, // QUANT_96
	{ 7, 0, 0 }, // QUANT_128
	{ 5, 0, 1 }, // QUANT_160
	{ 6, 1, 0 }, // QUANT_192
	{ 8, 0, 0 }  // QUANT_256
} };

/** @brief Packed trit values for each unpacked value, indexed [hi][][][][lo]. */
static const uint8_t integer_of_trits[3][3][3][3][3]{
	{
		{
			{
				{0, 1, 2},
				{4, 5, 6},
				{8, 9, 10}
			},
			{
				{16, 17, 18},
				{20, 21, 22},
				{24, 25, 26}
			},
			{
				{3, 7, 15},
				{19, 23, 27},
				{12, 13, 14}
			}
		},
		{
			{
				{32, 33, 34},
				{36, 37, 38},
				{40, 41, 42}
			},
			{
				{48, 49, 50},
				{52, 53, 54},
				{56, 57, 58}
			},
			{
				{35, 39, 47},
				{51, 55, 59},
				{44, 45, 46}
			}
		},
		{
			{
				{64, 65, 66},
				{68, 69, 70},
				{72, 73, 74}
			},
			{
				{80, 81, 82},
				{84, 85, 86},
				{88, 89, 90}
			},
			{
				{67, 71, 79},
				{83, 87, 91},
				{76, 77, 78}
			}
		}
	},
	{
		{
			{
				{128, 129, 130},
				{132, 133, 134},
				{136, 137, 138}
			},
			{
				{144, 145, 146},
				{148, 149, 150},
				{152, 153, 154}
			},
			{
				{131, 135, 143},
				{147, 151, 155},
				{140, 141, 142}
			}
		},
		{
			{
				{160, 161, 162},
				{164, 165, 166},
				{168, 169, 170}
			},
			{
				{176, 177, 178},
				{180, 181, 182},
				{184, 185, 186}
			},
			{
				{163, 167, 175},
				{179, 183, 187},
				{172, 173, 174}
			}
		},
		{
			{
				{192, 193, 194},
				{196, 197, 198},
				{200, 201, 202}
			},
			{
				{208, 209, 210},
				{212, 213, 214},
				{216, 217, 218}
			},
			{
				{195, 199, 207},
				{211, 215, 219},
				{204, 205, 206}
			}
		}
	},
	{
		{
			{
				{96, 97, 98},
				{100, 101, 102},
				{104, 105, 106}
			},
			{
				{112, 113, 114},
				{116, 117, 118},
				{120, 121, 122}
			},
			{
				{99, 103, 111},
				{115, 119, 123},
				{108, 109, 110}
			}
		},
		{
			{
				{224, 225, 226},
				{228, 229, 230},
				{232, 233, 234}
			},
			{
				{240, 241, 242},
				{244, 245, 246},
				{248, 249, 250}
			},
			{
				{227, 231, 239},
				{243, 247, 251},
				{236, 237, 238}
			}
		},
		{
			{
				{28, 29, 30},
				{60, 61, 62},
				{92, 93, 94}
			},
			{
				{156, 157, 158},
				{188, 189, 190},
				{220, 221, 222}
			},
			{
				{31, 63, 127},
				{159, 191, 255},
				{252, 253, 254}
			}
		}
	}
};

/** @brief Packed quint values for each unpacked value, indexed [hi][mid][lo]. */
static const uint8_t integer_of_quints[5][5][5]{
	{
		{0, 1, 2, 3, 4},
		{8, 9, 10, 11, 12},
		{16, 17, 18, 19, 20},
		{24, 25, 26, 27, 28},
		{5, 13, 21, 29, 6}
	},
	{
		{32, 33, 34, 35, 36},
		{40, 41, 42, 43, 44},
		{48, 49, 50, 51, 52},
		{56, 57, 58, 59, 60},
		{37, 45, 53, 61, 14}
	},
	{
		{64, 65, 66, 67, 68},
		{72, 73, 74, 75, 76},
		{80, 81, 82, 83, 84},
		{88, 89, 90, 91, 92},
		{69, 77, 85, 93, 22}
	},
	{
		{96, 97, 98, 99, 100},
		{104, 105, 106, 107, 108},
		{112, 113, 114, 115, 116},
		{120, 121, 122, 123, 124},
		{101, 109, 117, 125, 30}
	},
	{
		{102, 103, 70, 71, 38},
		{110, 111, 78, 79, 46},
		{118, 119, 86, 87, 54},
		{126, 127, 94, 95, 62},
		{39, 47, 55, 63, 31}
	}
};

struct transfer_table {
	uint8_t scramble_map[32];
};

static transfer_table transfet_tables[12] = {
	//QUANT_2
	{{0, 1}},
	//QUANT_3
	{{0, 1, 2}},
	//QUANT_4
	{{0, 1, 2, 3}},
	//QUANT_5
	{{0, 1, 2, 3, 4}},
	//QUANT_6
	{{0, 2, 4, 5, 3, 1}},
	//QUANT_8
	{{0, 1, 2, 3, 4, 5, 6, 7}},
	//QUANT_10
	{{0, 2, 4, 6, 8, 9, 7, 5, 3, 1}},
	//QUANT_12
	{{0, 4, 8, 2, 6, 10, 11, 7, 3, 9, 5, 1}},
	//QUANT_20
	{{0, 4, 8, 12, 16, 2, 6, 10, 14, 18, 19, 15, 11, 7, 3, 17, 13, 9, 5, 1}},
	//QUANT_20
	{{0, 8, 16, 2, 10, 18, 4, 12, 20, 6, 14, 22, 23, 15, 7, 21, 13, 5, 19, 11, 3, 17, 9, 1}},
	//QUANT_32
	{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}}
};

/**
 * @brief The precomputed table for packing quantized color values.
 *
 * Converts quant value in 0-255 range into packed quant value in 0-N range,
 * with BISE scrambling applied.
 *
 * Indexed by [quant_mode - 4][data_value].
 */
const uint8_t color_uquant_to_scrambled_pquant_tables[17][256]{
	{ // QUANT_6
		  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   2,   2,   2,   2,   2,   2,
		  2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
		  2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
		  2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   4,   4,   4,
		  4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,
		  4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,
		  4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,
		  5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,
		  5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,
		  5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,
		  5,   5,   5,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,
		  3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,
		  3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,
		  3,   3,   3,   3,   3,   3,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
		  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1
	},
	{ // QUANT_8
		  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
		  0,   0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
		  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
		  1,   1,   1,   1,   1,   1,   1,   2,   2,   2,   2,   2,   2,   2,   2,   2,
		  2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
		  2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   3,   3,   3,   3,   3,
		  3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,
		  3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,
		  4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,
		  4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,
		  4,   4,   4,   4,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,
		  5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,
		  5,   5,   5,   5,   5,   5,   5,   5,   5,   6,   6,   6,   6,   6,   6,   6,
		  6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,
		  6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   7,   7,   7,
		  7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7
	},
	{ // QUANT_10
		  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   2,   2,
		  2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
		  2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   4,   4,   4,   4,   4,   4,
		  4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,
		  4,   4,   4,   4,   4,   4,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,
		  6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,
		  6,   6,   6,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,
		  8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,
		  9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,
		  9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   7,   7,   7,
		  7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,
		  7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   5,   5,   5,   5,   5,   5,
		  5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,
		  5,   5,   5,   5,   5,   5,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,
		  3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,
		  3,   3,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1
	},
	{ // QUANT_12
		  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   4,   4,   4,   4,
		  4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,
		  4,   4,   4,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,
		  8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   2,   2,   2,   2,   2,   2,
		  2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
		  2,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,
		  6,   6,   6,   6,   6,   6,   6,   6,  10,  10,  10,  10,  10,  10,  10,  10,
		 10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,
		 11,  11,  11,  11,  11,  11,  11,  11,  11,  11,  11,  11,  11,  11,  11,  11,
		 11,  11,  11,  11,  11,  11,  11,  11,   7,   7,   7,   7,   7,   7,   7,   7,
		  7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   3,
		  3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,
		  3,   3,   3,   3,   3,   3,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,
		  9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   5,   5,   5,
		  5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,
		  5,   5,   5,   5,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1
	},
	{ // QUANT_16
		  0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   1,   1,   1,   1,   1,
		  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   2,   2,   2,   2,   2,   2,
		  2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   3,   3,   3,   3,   3,
		  3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   4,   4,   4,   4,
		  4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   5,   5,   5,
		  5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   6,   6,
		  6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   6,   7,
		  7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,   7,
		  8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,
		  8,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,
		  9,   9,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,
		 10,  10,  10,  11,  11,  11,  11,  11,  11,  11,  11,  11,  11,  11,  11,  11,
		 11,  11,  11,  11,  12,  12,  12,  12,  12,  12,  12,  12,  12,  12,  12,  12,
		 12,  12,  12,  12,  12,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13,
		 13,  13,  13,  13,  13,  13,  14,  14,  14,  14,  14,  14,  14,  14,  14,  14,
		 14,  14,  14,  14,  14,  14,  14,  15,  15,  15,  15,  15,  15,  15,  15,  15
	},
	{ // QUANT_20
		  0,   0,   0,   0,   0,   0,   0,   4,   4,   4,   4,   4,   4,   4,   4,   4,
		  4,   4,   4,   4,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,
		  8,   8,  12,  12,  12,  12,  12,  12,  12,  12,  12,  12,  12,  12,  12,  16,
		 16,  16,  16,  16,  16,  16,  16,  16,  16,  16,  16,  16,  16,   2,   2,   2,
		  2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   6,   6,   6,   6,   6,   6,
		  6,   6,   6,   6,   6,   6,   6,  10,  10,  10,  10,  10,  10,  10,  10,  10,
		 10,  10,  10,  10,  10,  14,  14,  14,  14,  14,  14,  14,  14,  14,  14,  14,
		 14,  14,  18,  18,  18,  18,  18,  18,  18,  18,  18,  18,  18,  18,  18,  18,
		 19,  19,  19,  19,  19,  19,  19,  19,  19,  19,  19,  19,  19,  19,  15,  15,
		 15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  15,  11,  11,  11,  11,  11,
		 11,  11,  11,  11,  11,  11,  11,  11,  11,   7,   7,   7,   7,   7,   7,   7,
		  7,   7,   7,   7,   7,   7,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,
		  3,   3,   3,  17,  17,  17,  17,  17,  17,  17,  17,  17,  17,  17,  17,  17,
		 17,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13,   9,   9,
		  9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   5,   5,   5,   5,
		  5,   5,   5,   5,   5,   5,   5,   5,   5,   1,   1,   1,   1,   1,   1,   1
	},
	{ // QUANT_24
		  0,   0,   0,   0,   0,   0,   8,   8,   8,   8,   8,   8,   8,   8,   8,   8,
		  8,  16,  16,  16,  16,  16,  16,  16,  16,  16,  16,  16,   2,   2,   2,   2,
		  2,   2,   2,   2,   2,   2,   2,  10,  10,  10,  10,  10,  10,  10,  10,  10,
		 10,  10,  18,  18,  18,  18,  18,  18,  18,  18,  18,  18,  18,   4,   4,   4,
		  4,   4,   4,   4,   4,   4,   4,   4,  12,  12,  12,  12,  12,  12,  12,  12,
		 12,  12,  12,  20,  20,  20,  20,  20,  20,  20,  20,  20,  20,  20,   6,   6,
		  6,   6,   6,   6,   6,   6,   6,   6,   6,  14,  14,  14,  14,  14,  14,  14,
		 14,  14,  14,  14,  22,  22,  22,  22,  22,  22,  22,  22,  22,  22,  22,  22,
		 23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  23,  15,  15,  15,  15,
		 15,  15,  15,  15,  15,  15,  15,   7,   7,   7,   7,   7,   7,   7,   7,   7,
		  7,   7,  21,  21,  21,  21,  21,  21,  21,  21,  21,  21,  21,  13,  13,  13,
		 13,  13,  13,  13,  13,  13,  13,  13,   5,   5,   5,   5,   5,   5,   5,   5,
		  5,   5,   5,  19,  19,  19,  19,  19,  19,  19,  19,  19,  19,  19,  11,  11,
		 11,  11,  11,  11,  11,  11,  11,  11,  11,   3,   3,   3,   3,   3,   3,   3,
		  3,   3,   3,   3,  17,  17,  17,  17,  17,  17,  17,  17,  17,  17,  17,   9,
		  9,   9,   9,   9,   9,   9,   9,   9,   9,   9,   1,   1,   1,   1,   1,   1
	},
	{ // QUANT_32
		  0,   0,   0,   0,   1,   1,   1,   1,   1,   1,   1,   1,   2,   2,   2,   2,
		  2,   2,   2,   2,   3,   3,   3,   3,   3,   3,   3,   3,   3,   4,   4,   4,
		  4,   4,   4,   4,   4,   5,   5,   5,   5,   5,   5,   5,   5,   6,   6,   6,
		  6,   6,   6,   6,   6,   7,   7,   7,   7,   7,   7,   7,   7,   7,   8,   8,
		  8,   8,   8,   8,   8,   8,   9,   9,   9,   9,   9,   9,   9,   9,  10,  10,
		 10,  10,  10,  10,  10,  10,  11,  11,  11,  11,  11,  11,  11,  11,  11,  12,
		 12,  12,  12,  12,  12,  12,  12,  13,  13,  13,  13,  13,  13,  13,  13,  14,
		 14,  14,  14,  14,  14,  14,  14,  15,  15,  15,  15,  15,  15,  15,  15,  15,
		 16,  16,  16,  16,  16,  16,  16,  16,  17,  17,  17,  17,  17,  17,  17,  17,
		 18,  18,  18,  18,  18,  18,  18,  18,  19,  19,  19,  19,  19,  19,  19,  19,
		 19,  20,  20,  20,  20,  20,  20,  20,  20,  21,  21,  21,  21,  21,  21,  21,
		 21,  22,  22,  22,  22,  22,  22,  22,  22,  23,  23,  23,  23,  23,  23,  23,
		 23,  23,  24,  24,  24,  24,  24,  24,  24,  24,  25,  25,  25,  25,  25,  25,
		 25,  25,  26,  26,  26,  26,  26,  26,  26,  26,  27,  27,  27,  27,  27,  27,
		 27,  27,  27,  28,  28,  28,  28,  28,  28,  28,  28,  29,  29,  29,  29,  29,
		 29,  29,  29,  30,  30,  30,  30,  30,  30,  30,  30,  31,  31,  31,  31,  31
	},
	{ // QUANT_40
		  0,   0,   0,   8,   8,   8,   8,   8,   8,   8,  16,  16,  16,  16,  16,  16,
		 24,  24,  24,  24,  24,  24,  24,  32,  32,  32,  32,  32,  32,  32,   2,   2,
		  2,   2,   2,   2,  10,  10,  10,  10,  10,  10,  18,  18,  18,  18,  18,  18,
		 18,  26,  26,  26,  26,  26,  26,  34,  34,  34,  34,  34,  34,  34,   4,   4,
		  4,   4,   4,   4,  12,  12,  12,  12,  12,  12,  12,  20,  20,  20,  20,  20,
		 20,  28,  28,  28,  28,  28,  28,  28,  36,  36,  36,  36,  36,  36,  36,   6,
		  6,   6,   6,   6,   6,  14,  14,  14,  14,  14,  14,  22,  22,  22,  22,  22,
		 22,  22,  30,  30,  30,  30,  30,  30,  38,  38,  38,  38,  38,  38,  38,  38,
		 39,  39,  39,  39,  39,  39,  39,  39,  31,  31,  31,  31,  31,  31,  23,  23,
		 23,  23,  23,  23,  23,  15,  15,  15,  15,  15,  15,   7,   7,   7,   7,   7,
		  7,  37,  37,  37,  37,  37,  37,  37,  29,  29,  29,  29,  29,  29,  29,  21,
		 21,  21,  21,  21,  21,  13,  13,  13,  13,  13,  13,  13,   5,   5,   5,   5,
		  5,   5,  35,  35,  35,  35,  35,  35,  35,  27,  27,  27,  27,  27,  27,  19,
		 19,  19,  19,  19,  19,  19,  11,  11,  11,  11,  11,  11,   3,   3,   3,   3,
		  3,   3,  33,  33,  33,  33,  33,  33,  33,  25,  25,  25,  25,  25,  25,  25,
		 17,  17,  17,  17,  17,  17,   9,   9,   9,   9,   9,   9,   9,   1,   1,   1
	},
	{ // QUANT_48
		  0,   0,   0,  16,  16,  16,  16,  16,  32,  32,  32,  32,  32,  32,   2,   2,
		  2,   2,   2,  18,  18,  18,  18,  18,  34,  34,  34,  34,  34,  34,   4,   4,
		  4,   4,   4,  20,  20,  20,  20,  20,  20,  36,  36,  36,  36,  36,   6,   6,
		  6,   6,   6,  22,  22,  22,  22,  22,  22,  38,  38,  38,  38,  38,  38,   8,
		  8,   8,   8,   8,  24,  24,  24,  24,  24,  40,  40,  40,  40,  40,  40,  10,
		 10,  10,  10,  10,  26,  26,  26,  26,  26,  42,  42,  42,  42,  42,  42,  12,
		 12,  12,  12,  12,  28,  28,  28,  28,  28,  28,  44,  44,  44,  44,  44,  14,
		 14,  14,  14,  14,  30,  30,  30,  30,  30,  30,  46,  46,  46,  46,  46,  46,
		 47,  47,  47,  47,  47,  47,  31,  31,  31,  31,  31,  31,  15,  15,  15,  15,
		 15,  45,  45,  45,  45,  45,  29,  29,  29,  29,  29,  29,  13,  13,  13,  13,
		 13,  43,  43,  43,  43,  43,  43,  27,  27,  27,  27,  27,  11,  11,  11,  11,
		 11,  41,  41,  41,  41,  41,  41,  25,  25,  25,  25,  25,   9,   9,   9,   9,
		  9,  39,  39,  39,  39,  39,  39,  23,  23,  23,  23,  23,  23,   7,   7,   7,
		  7,   7,  37,  37,  37,  37,  37,  21,  21,  21,  21,  21,  21,   5,   5,   5,
		  5,   5,  35,  35,  35,  35,  35,  35,  19,  19,  19,  19,  19,   3,   3,   3,
		  3,   3,  33,  33,  33,  33,  33,  33,  17,  17,  17,  17,  17,   1,   1,   1
	},
	{ // QUANT_64
		  0,   0,   1,   1,   1,   1,   2,   2,   2,   2,   3,   3,   3,   3,   4,   4,
		  4,   4,   5,   5,   5,   5,   6,   6,   6,   6,   7,   7,   7,   7,   8,   8,
		  8,   8,   9,   9,   9,   9,  10,  10,  10,  10,  11,  11,  11,  11,  12,  12,
		 12,  12,  13,  13,  13,  13,  14,  14,  14,  14,  15,  15,  15,  15,  15,  16,
		 16,  16,  16,  17,  17,  17,  17,  18,  18,  18,  18,  19,  19,  19,  19,  20,
		 20,  20,  20,  21,  21,  21,  21,  22,  22,  22,  22,  23,  23,  23,  23,  24,
		 24,  24,  24,  25,  25,  25,  25,  26,  26,  26,  26,  27,  27,  27,  27,  28,
		 28,  28,  28,  29,  29,  29,  29,  30,  30,  30,  30,  31,  31,  31,  31,  31,
		 32,  32,  32,  32,  33,  33,  33,  33,  34,  34,  34,  34,  35,  35,  35,  35,
		 36,  36,  36,  36,  37,  37,  37,  37,  38,  38,  38,  38,  39,  39,  39,  39,
		 40,  40,  40,  40,  41,  41,  41,  41,  42,  42,  42,  42,  43,  43,  43,  43,
		 44,  44,  44,  44,  45,  45,  45,  45,  46,  46,  46,  46,  47,  47,  47,  47,
		 47,  48,  48,  48,  48,  49,  49,  49,  49,  50,  50,  50,  50,  51,  51,  51,
		 51,  52,  52,  52,  52,  53,  53,  53,  53,  54,  54,  54,  54,  55,  55,  55,
		 55,  56,  56,  56,  56,  57,  57,  57,  57,  58,  58,  58,  58,  59,  59,  59,
		 59,  60,  60,  60,  60,  61,  61,  61,  61,  62,  62,  62,  62,  63,  63,  63
	},
	{ // QUANT_80
		  0,   0,  16,  16,  16,  32,  32,  32,  48,  48,  48,  64,  64,  64,  64,   2,
		  2,   2,  18,  18,  18,  34,  34,  34,  50,  50,  50,  66,  66,  66,  66,   4,
		  4,   4,  20,  20,  20,  36,  36,  36,  52,  52,  52,  52,  68,  68,  68,   6,
		  6,   6,  22,  22,  22,  38,  38,  38,  54,  54,  54,  54,  70,  70,  70,   8,
		  8,   8,  24,  24,  24,  40,  40,  40,  40,  56,  56,  56,  72,  72,  72,  10,
		 10,  10,  26,  26,  26,  42,  42,  42,  42,  58,  58,  58,  74,  74,  74,  12,
		 12,  12,  28,  28,  28,  28,  44,  44,  44,  60,  60,  60,  76,  76,  76,  14,
		 14,  14,  30,  30,  30,  30,  46,  46,  46,  62,  62,  62,  78,  78,  78,  78,
		 79,  79,  79,  79,  63,  63,  63,  47,  47,  47,  31,  31,  31,  31,  15,  15,
		 15,  77,  77,  77,  61,  61,  61,  45,  45,  45,  29,  29,  29,  29,  13,  13,
		 13,  75,  75,  75,  59,  59,  59,  43,  43,  43,  43,  27,  27,  27,  11,  11,
		 11,  73,  73,  73,  57,  57,  57,  41,  41,  41,  41,  25,  25,  25,   9,   9,
		  9,  71,  71,  71,  55,  55,  55,  55,  39,  39,  39,  23,  23,  23,   7,   7,
		  7,  69,  69,  69,  53,  53,  53,  53,  37,  37,  37,  21,  21,  21,   5,   5,
		  5,  67,  67,  67,  67,  51,  51,  51,  35,  35,  35,  19,  19,  19,   3,   3,
		  3,  65,  65,  65,  65,  49,  49,  49,  33,  33,  33,  17,  17,  17,   1,   1
	},
	{ // QUANT_96
		  0,  32,  32,  32,  64,  64,  64,   2,   2,  34,  34,  34,  66,  66,  66,   4,
		  4,  36,  36,  36,  68,  68,  68,   6,   6,  38,  38,  38,  70,  70,  70,   8,
		  8,   8,  40,  40,  72,  72,  72,  10,  10,  10,  42,  42,  74,  74,  74,  12,
		 12,  12,  44,  44,  76,  76,  76,  14,  14,  14,  46,  46,  78,  78,  78,  16,
		 16,  16,  48,  48,  48,  80,  80,  80,  18,  18,  50,  50,  50,  82,  82,  82,
		 20,  20,  52,  52,  52,  84,  84,  84,  22,  22,  54,  54,  54,  86,  86,  86,
		 24,  24,  56,  56,  56,  88,  88,  88,  26,  26,  58,  58,  58,  90,  90,  90,
		 28,  28,  60,  60,  60,  92,  92,  92,  30,  30,  62,  62,  62,  94,  94,  94,
		 95,  95,  95,  63,  63,  63,  31,  31,  93,  93,  93,  61,  61,  61,  29,  29,
		 91,  91,  91,  59,  59,  59,  27,  27,  89,  89,  89,  57,  57,  57,  25,  25,
		 87,  87,  87,  55,  55,  55,  23,  23,  85,  85,  85,  53,  53,  53,  21,  21,
		 83,  83,  83,  51,  51,  51,  19,  19,  81,  81,  81,  49,  49,  49,  17,  17,
		 17,  79,  79,  79,  47,  47,  15,  15,  15,  77,  77,  77,  45,  45,  13,  13,
		 13,  75,  75,  75,  43,  43,  11,  11,  11,  73,  73,  73,  41,  41,   9,   9,
		  9,  71,  71,  71,  39,  39,  39,   7,   7,  69,  69,  69,  37,  37,  37,   5,
		  5,  67,  67,  67,  35,  35,  35,   3,   3,  65,  65,  65,  33,  33,  33,   1
	},
	{ // QUANT_128
		  0,   1,   1,   2,   2,   3,   3,   4,   4,   5,   5,   6,   6,   7,   7,   8,
		  8,   9,   9,  10,  10,  11,  11,  12,  12,  13,  13,  14,  14,  15,  15,  16,
		 16,  17,  17,  18,  18,  19,  19,  20,  20,  21,  21,  22,  22,  23,  23,  24,
		 24,  25,  25,  26,  26,  27,  27,  28,  28,  29,  29,  30,  30,  31,  31,  32,
		 32,  33,  33,  34,  34,  35,  35,  36,  36,  37,  37,  38,  38,  39,  39,  40,
		 40,  41,  41,  42,  42,  43,  43,  44,  44,  45,  45,  46,  46,  47,  47,  48,
		 48,  49,  49,  50,  50,  51,  51,  52,  52,  53,  53,  54,  54,  55,  55,  56,
		 56,  57,  57,  58,  58,  59,  59,  60,  60,  61,  61,  62,  62,  63,  63,  63,
		 64,  64,  65,  65,  66,  66,  67,  67,  68,  68,  69,  69,  70,  70,  71,  71,
		 72,  72,  73,  73,  74,  74,  75,  75,  76,  76,  77,  77,  78,  78,  79,  79,
		 80,  80,  81,  81,  82,  82,  83,  83,  84,  84,  85,  85,  86,  86,  87,  87,
		 88,  88,  89,  89,  90,  90,  91,  91,  92,  92,  93,  93,  94,  94,  95,  95,
		 96,  96,  97,  97,  98,  98,  99,  99, 100, 100, 101, 101, 102, 102, 103, 103,
		104, 104, 105, 105, 106, 106, 107, 107, 108, 108, 109, 109, 110, 110, 111, 111,
		112, 112, 113, 113, 114, 114, 115, 115, 116, 116, 117, 117, 118, 118, 119, 119,
		120, 120, 121, 121, 122, 122, 123, 123, 124, 124, 125, 125, 126, 126, 127, 127
	},
	{ // QUANT_160
		  0,  32,  64,  64,  96, 128, 128, 128,   2,  34,  66,  66,  98, 130, 130, 130,
		  4,  36,  68,  68, 100, 132, 132, 132,   6,  38,  70,  70, 102, 134, 134, 134,
		  8,  40,  72,  72, 104, 136, 136, 136,  10,  42,  74,  74, 106, 138, 138, 138,
		 12,  44,  76,  76, 108, 140, 140, 140,  14,  46,  78,  78, 110, 142, 142, 142,
		 16,  48,  80,  80, 112, 144, 144, 144,  18,  50,  82,  82, 114, 146, 146, 146,
		 20,  52,  84,  84, 116, 148, 148, 148,  22,  54,  86,  86, 118, 150, 150, 150,
		 24,  56,  88,  88, 120, 152, 152, 152,  26,  58,  90,  90, 122, 154, 154, 154,
		 28,  60,  92,  92, 124, 156, 156, 156,  30,  62,  94,  94, 126, 158, 158, 158,
		159, 159, 159, 127,  95,  95,  63,  31, 157, 157, 157, 125,  93,  93,  61,  29,
		155, 155, 155, 123,  91,  91,  59,  27, 153, 153, 153, 121,  89,  89,  57,  25,
		151, 151, 151, 119,  87,  87,  55,  23, 149, 149, 149, 117,  85,  85,  53,  21,
		147, 147, 147, 115,  83,  83,  51,  19, 145, 145, 145, 113,  81,  81,  49,  17,
		143, 143, 143, 111,  79,  79,  47,  15, 141, 141, 141, 109,  77,  77,  45,  13,
		139, 139, 139, 107,  75,  75,  43,  11, 137, 137, 137, 105,  73,  73,  41,   9,
		135, 135, 135, 103,  71,  71,  39,   7, 133, 133, 133, 101,  69,  69,  37,   5,
		131, 131, 131,  99,  67,  67,  35,   3, 129, 129, 129,  97,  65,  65,  33,   1
	},
	{ // QUANT_192
		  0,  64, 128, 128,   2,  66, 130, 130,   4,  68, 132, 132,   6,  70, 134, 134,
		  8,  72, 136, 136,  10,  74, 138, 138,  12,  76, 140, 140,  14,  78, 142, 142,
		 16,  80, 144, 144,  18,  82, 146, 146,  20,  84, 148, 148,  22,  86, 150, 150,
		 24,  88, 152, 152,  26,  90, 154, 154,  28,  92, 156, 156,  30,  94, 158, 158,
		 32,  96, 160, 160,  34,  98, 162, 162,  36, 100, 164, 164,  38, 102, 166, 166,
		 40, 104, 168, 168,  42, 106, 170, 170,  44, 108, 172, 172,  46, 110, 174, 174,
		 48, 112, 176, 176,  50, 114, 178, 178,  52, 116, 180, 180,  54, 118, 182, 182,
		 56, 120, 184, 184,  58, 122, 186, 186,  60, 124, 188, 188,  62, 126, 190, 190,
		191, 191, 127,  63, 189, 189, 125,  61, 187, 187, 123,  59, 185, 185, 121,  57,
		183, 183, 119,  55, 181, 181, 117,  53, 179, 179, 115,  51, 177, 177, 113,  49,
		175, 175, 111,  47, 173, 173, 109,  45, 171, 171, 107,  43, 169, 169, 105,  41,
		167, 167, 103,  39, 165, 165, 101,  37, 163, 163,  99,  35, 161, 161,  97,  33,
		159, 159,  95,  31, 157, 157,  93,  29, 155, 155,  91,  27, 153, 153,  89,  25,
		151, 151,  87,  23, 149, 149,  85,  21, 147, 147,  83,  19, 145, 145,  81,  17,
		143, 143,  79,  15, 141, 141,  77,  13, 139, 139,  75,  11, 137, 137,  73,   9,
		135, 135,  71,   7, 133, 133,  69,   5, 131, 131,  67,   3, 129, 129,  65,   1
	},
	{ // QUANT_256
		  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,
		 16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
		 32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
		 48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,
		 64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
		 80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
		 96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
		112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
		128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
		144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
		160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
		176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
		192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
		208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
		224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
		240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255
	}
};

struct ise_size
{
	uint8_t scale : 6;
	uint8_t divisor : 2;
};


static const std::array<ise_size, 21> ise_sizes{ {
	{  1, 0 }, // QUANT_2
	{  8, 2 }, // QUANT_3
	{  2, 0 }, // QUANT_4
	{  7, 1 }, // QUANT_5
	{ 13, 2 }, // QUANT_6
	{  3, 0 }, // QUANT_8
	{ 10, 1 }, // QUANT_10
	{ 18, 2 }, // QUANT_12
	{  4, 0 }, // QUANT_16
	{ 13, 1 }, // QUANT_20
	{ 23, 2 }, // QUANT_24
	{  5, 0 }, // QUANT_32
	{ 16, 1 }, // QUANT_40
	{ 28, 2 }, // QUANT_48
	{  6, 0 }, // QUANT_64
	{ 19, 1 }, // QUANT_80
	{ 33, 2 }, // QUANT_96
	{  7, 0 }, // QUANT_128
	{ 22, 1 }, // QUANT_160
	{ 38, 2 }, // QUANT_192
	{  8, 0 }  // QUANT_256
} };


unsigned int get_ise_sequence_bitcount(
	unsigned int character_count,
	quant_method quant_level
) {
	// Cope with out-of bounds values - input might be invalid
	if (static_cast<size_t>(quant_level) >= ise_sizes.size())
	{
		// Arbitrary large number that's more than an ASTC block can hold
		return 1024;
	}

	auto& entry = ise_sizes[quant_level];
	unsigned int divisor = (entry.divisor << 1) + 1;
	return (entry.scale * character_count + divisor - 1) / divisor;
}

/**
 * @brief Write up to 8 bits at an arbitrary bit offset.
 *
 * The stored value is at most 8 bits, but can be stored at an offset of between 0 and 7 bits so may
 * span two separate bytes in memory.
 *
 * @param         value       The value to write.
 * @param         bitcount    The number of bits to write, starting from LSB.
 * @param         bitoffset   The bit offset to store at, between 0 and 7.
 * @param[in,out] ptr         The data pointer to write to.
 */
static inline void write_bits(
	unsigned int value,
	unsigned int bitcount,
	unsigned int bitoffset,
	uint8_t ptr[2]
) {
	unsigned int mask = (1 << bitcount) - 1;
	value &= mask;
	ptr += bitoffset >> 3;
	bitoffset &= 7;
	value <<= bitoffset;
	mask <<= bitoffset;
	mask = ~mask;

	ptr[0] &= mask;
	ptr[0] |= value;
	ptr[1] &= mask >> 8;
	ptr[1] |= value >> 8;
}

void encode_ise(
	quant_method quant_level,
	unsigned int character_count,
	const uint8_t* input_data,
	uint8_t* output_data,
	unsigned int bit_offset
) {
	
	unsigned int bits = btq_counts[quant_level].bits;
	unsigned int trits = btq_counts[quant_level].trits;
	unsigned int quints = btq_counts[quant_level].quints;
	unsigned int mask = (1 << bits) - 1;

	// Write out trits and bits
	if (trits)
	{
		unsigned int i = 0;
		unsigned int full_trit_blocks = character_count / 5;

		for (unsigned int j = 0; j < full_trit_blocks; j++)
		{
			unsigned int i4 = input_data[i + 4] >> bits;
			unsigned int i3 = input_data[i + 3] >> bits;
			unsigned int i2 = input_data[i + 2] >> bits;
			unsigned int i1 = input_data[i + 1] >> bits;
			unsigned int i0 = input_data[i + 0] >> bits;

			uint8_t T = integer_of_trits[i4][i3][i2][i1][i0];

			// The max size of a trit bit count is 6, so we can always safely
			// pack a single MX value with the following 1 or 2 T bits.
			uint8_t pack;

			// Element 0 + T0 + T1
			pack = (input_data[i++] & mask) | (((T >> 0) & 0x3) << bits);
			write_bits(pack, bits + 2, bit_offset, output_data);
			bit_offset += bits + 2;

			// Element 1 + T2 + T3
			pack = (input_data[i++] & mask) | (((T >> 2) & 0x3) << bits);
			write_bits(pack, bits + 2, bit_offset, output_data);
			bit_offset += bits + 2;

			// Element 2 + T4
			pack = (input_data[i++] & mask) | (((T >> 4) & 0x1) << bits);
			write_bits(pack, bits + 1, bit_offset, output_data);
			bit_offset += bits + 1;

			// Element 3 + T5 + T6
			pack = (input_data[i++] & mask) | (((T >> 5) & 0x3) << bits);
			write_bits(pack, bits + 2, bit_offset, output_data);
			bit_offset += bits + 2;

			// Element 4 + T7
			pack = (input_data[i++] & mask) | (((T >> 7) & 0x1) << bits);
			write_bits(pack, bits + 1, bit_offset, output_data);
			bit_offset += bits + 1;
		}

		// Loop tail for a partial block
		if (i != character_count)
		{
			// i4 cannot be present - we know the block is partial
			// i0 must be present - we know the block isn't empty
			unsigned int i4 = 0;
			unsigned int i3 = i + 3 >= character_count ? 0 : input_data[i + 3] >> bits;
			unsigned int i2 = i + 2 >= character_count ? 0 : input_data[i + 2] >> bits;
			unsigned int i1 = i + 1 >= character_count ? 0 : input_data[i + 1] >> bits;
			unsigned int i0 = input_data[i + 0] >> bits;

			uint8_t T = integer_of_trits[i4][i3][i2][i1][i0];

			for (unsigned int j = 0; i < character_count; i++, j++)
			{
				// Truncated table as this iteration is always partital
				static const uint8_t tbits[4]{ 2, 2, 1, 2 };
				static const uint8_t tshift[4]{ 0, 2, 4, 5 };

				uint8_t pack = (input_data[i] & mask) |
					(((T >> tshift[j]) & ((1 << tbits[j]) - 1)) << bits);

				write_bits(pack, bits + tbits[j], bit_offset, output_data);
				bit_offset += bits + tbits[j];
			}
		}
	}
	// Write out quints and bits
	else if (quints)
	{
		unsigned int i = 0;
		unsigned int full_quint_blocks = character_count / 3;

		for (unsigned int j = 0; j < full_quint_blocks; j++)
		{
			unsigned int i2 = input_data[i + 2] >> bits;
			unsigned int i1 = input_data[i + 1] >> bits;
			unsigned int i0 = input_data[i + 0] >> bits;

			uint8_t T = integer_of_quints[i2][i1][i0];

			// The max size of a quint bit count is 5, so we can always safely
			// pack a single M value with the following 2 or 3 T bits.
			uint8_t pack;

			// Element 0
			pack = (input_data[i++] & mask) | (((T >> 0) & 0x7) << bits);
			write_bits(pack, bits + 3, bit_offset, output_data);
			bit_offset += bits + 3;

			// Element 1
			pack = (input_data[i++] & mask) | (((T >> 3) & 0x3) << bits);
			write_bits(pack, bits + 2, bit_offset, output_data);
			bit_offset += bits + 2;

			// Element 2
			pack = (input_data[i++] & mask) | (((T >> 5) & 0x3) << bits);
			write_bits(pack, bits + 2, bit_offset, output_data);
			bit_offset += bits + 2;
		}

		// Loop tail for a partial block
		if (i != character_count)
		{
			// i2 cannot be present - we know the block is partial
			// i0 must be present - we know the block isn't empty
			unsigned int i2 = 0;
			unsigned int i1 = i + 1 >= character_count ? 0 : input_data[i + 1] >> bits;
			unsigned int i0 = input_data[i + 0] >> bits;

			uint8_t T = integer_of_quints[i2][i1][i0];

			for (unsigned int j = 0; i < character_count; i++, j++)
			{
				// Truncated table as this iteration is always partital
				static const uint8_t tbits[2]{ 3, 2 };
				static const uint8_t tshift[2]{ 0, 3 };

				uint8_t pack = (input_data[i] & mask) |
					(((T >> tshift[j]) & ((1 << tbits[j]) - 1)) << bits);

				write_bits(pack, bits + tbits[j], bit_offset, output_data);
				bit_offset += bits + tbits[j];
			}
		}
	}
	// Write out just bits
	else
	{
		for (unsigned int i = 0; i < character_count; i++)
		{
			write_bits(input_data[i], bits, bit_offset, output_data);
			bit_offset += bits;
		}
	}
}

/**
 * @brief Reverse bits in a byte.
 *
 * @param p   The value to reverse.
  *
 * @return The reversed result.
 */
static inline int bitrev8(int p)
{
	p = ((p & 0x0F) << 4) | ((p >> 4) & 0x0F);
	p = ((p & 0x33) << 2) | ((p >> 2) & 0x33);
	p = ((p & 0x55) << 1) | ((p >> 1) & 0x55);
	return p;
}

void symbolic_to_physical(
	const block_descriptor& block_descriptor,
	const SymbolicBlock& symbolic_compressed_block,
	uint8_t physical_compressed_block[16]
) {

	//TODO: constant color block packing

	unsigned int partition_count = symbolic_compressed_block.partition_count;

	//compress weights
	//encoded as an ordianry integer sequence and then bit-reversed
	uint8_t weightbuf[16]{ 0 };

	block_mode bm = block_descriptor.block_modes[symbolic_compressed_block.block_mode_index];
	decimation_info di = block_descriptor.decimation_info_metadata[bm.decimation_mode];
	int weight_count = di.weight_count;
	quant_method weight_quant_method = static_cast<quant_method>(bm.quant_mode);
	float weight_qunat_levels = static_cast<float>(get_quant_level(weight_quant_method));
	int is_dual_plane = bm.is_dual_plane;

	int real_weight_count = is_dual_plane ? 2 * weight_count : weight_count;

	const auto& weight_scramble_map = transfet_tables[weight_quant_method].scramble_map;

	int bits_for_weights = get_ise_sequence_bitcount(real_weight_count, weight_quant_method);

	uint8_t weights[64];
	if (is_dual_plane) {
		for (int i = 0; i < weight_count; i++) {
			float uqw = static_cast<float>(symbolic_compressed_block.quantized_weights[i]);
			float qw = (uqw / 64.0f) * (weight_qunat_levels - 1.0f);
			int qwi = static_cast<int>(qw + 0.5f);
			weights[2 * i] = weight_scramble_map[qwi];

			uqw = static_cast<float>(symbolic_compressed_block.quantized_weights[i + WEIGHTS_PLANE2_OFFSET]);
			qw = (uqw / 64.0f) * (weight_qunat_levels - 1.0f);
			qwi = static_cast<int>(qw + 0.5f);
			weights[2 * i + 1] = weight_scramble_map[qwi];
		}
	}
	else {
		for (int i = 0; i < weight_count; i++) {
			float uqw = static_cast<float>(symbolic_compressed_block.quantized_weights[i]);
			float qw = (uqw / 64.0f) * (weight_qunat_levels - 1.0f);
			int qwi = static_cast<int>(qw + 0.5f);
			weights[i] = weight_scramble_map[qwi];
		}
	}

	encode_ise(weight_quant_method, real_weight_count, weights, weightbuf, 0);

	for (int i = 0; i < 16; i++) {
		physical_compressed_block[i] = static_cast<uint8_t>(bitrev8(weightbuf[15 - i]));
	}

	write_bits(symbolic_compressed_block.block_mode_index, 11, 0, physical_compressed_block);
	write_bits(partition_count - 1, 2, 11, physical_compressed_block);

	int below_weights_pos = 128 - bits_for_weights;

	if (partition_count > 1) {
		write_bits(symbolic_compressed_block.partition_index, 6, 13, physical_compressed_block);
		write_bits(symbolic_compressed_block.partition_index >> 6, PARTITION_INDEX_BITS - 6, 19, physical_compressed_block);

		if (symbolic_compressed_block.partition_formats_matched) {
			write_bits(symbolic_compressed_block.partition_formats[0] << 2, 6, 13 + PARTITION_INDEX_BITS, physical_compressed_block);
		}
		else {
			// Check endpoint types for each partition to determine the lowest class present
			int low_class = 4;

			for (unsigned int i = 0; i < partition_count; i++)
			{
				int class_of_format = symbolic_compressed_block.partition_formats[i] >> 2;
				low_class = std::min(class_of_format, low_class);
			}

			if (low_class == 3) {
				low_class = 2;
			}

			int encoded_type = low_class + 1;
			int bitpos = 2;

			for (unsigned int i = 0; i < partition_count; i++)
			{
				int classbit_of_format = (symbolic_compressed_block.partition_formats[i] >> 2) - low_class;
				encoded_type |= classbit_of_format << bitpos;
				bitpos++;
			}

			for (unsigned int i = 0; i < partition_count; i++)
			{
				int lowbits_of_format = symbolic_compressed_block.partition_formats[i] & 3;
				encoded_type |= lowbits_of_format << bitpos;
				bitpos += 2;
			}

			int encoded_type_lowpart = encoded_type & 0x3F;
			int encoded_type_highpart = encoded_type >> 6;
			int encoded_type_highpart_size = (3 * partition_count) - 4;
			int encoded_type_highpart_pos = 128 - bits_for_weights - encoded_type_highpart_size;
			write_bits(encoded_type_lowpart, 6, 13 + PARTITION_INDEX_BITS, physical_compressed_block);
			write_bits(encoded_type_highpart, encoded_type_highpart_size, encoded_type_highpart_pos, physical_compressed_block);
			below_weights_pos -= encoded_type_highpart_size;
		}
	}
	else {
		write_bits(symbolic_compressed_block.partition_formats[0], 4, 13, physical_compressed_block);
	}

	// In dual-plane mode, encode the color component of the second plane of weights
	if (is_dual_plane)
	{	
		//skip daul plane components, since they ate not implemented
		//write_bits(scb.plane2_component, 2, below_weights_pos - 2, pcb);
	}

	// Encode the color components
	uint8_t values_to_encode[32];
	int valuecount_to_encode = 0;

	const uint8_t* pack_table = color_uquant_to_scrambled_pquant_tables[symbolic_compressed_block.quant_mode - QUANT_6];
	for (unsigned int i = 0; i < symbolic_compressed_block.partition_count; i++) {
		int vals = 2 * (symbolic_compressed_block.partition_formats[i] >> 2) + 2;
		assert(vals <= 8);
		for (int j = 0; j < vals; j++)
		{
			values_to_encode[j + valuecount_to_encode] = pack_table[symbolic_compressed_block.packed_color_values[i * 8 + j]];
		}
		valuecount_to_encode += vals;
	}

	encode_ise(static_cast<quant_method>(symbolic_compressed_block.quant_mode), valuecount_to_encode,
		values_to_encode, physical_compressed_block, symbolic_compressed_block.partition_count == 1 ? 17 : 19 + PARTITION_INDEX_BITS);
}