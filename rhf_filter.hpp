#pragma once

struct RHFParameters
{
	RHFParameters() : patch_size(1), search_size(6), nscales(2), knn(2), threshold(1)
	{
	}

	int patch_size; // Half the patch size
	int search_size; // Half the search window size
	float threshold; // Distance threshold
	int nscales; // Number of scales
	int knn; // Minimum number of similar patches
};

// channel_histograms[0] is reserved for total number of samples per pixel
void rhf_filter(int width, int height, int bins, const float* input_image, const float* channel_histograms[], float* out_image, const RHFParameters& params = RHFParameters());
