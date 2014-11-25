#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <tuple>
#include <vector>

#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>
#include <tbb/reader_writer_lock.h>

#include <atomic>

#include "rhf_filter.hpp"

struct float4
{
	float x, y, z, w;
};

float chi2_distance(const float* x_hist, const float* y_hist, const float nsx, const float nsy, int numbins)
{
	float dist = 0;

	int k = 0;

	const float k0 = std::sqrt(nsy / nsx);
	const float k1 = std::sqrt(nsx / nsy);

	for (int i = 0; i < numbins; i++)
	{
		if (x_hist[i] + y_hist[i] > 0)
		{
			float t = k0 * x_hist[i] - k1 * y_hist[i];

			dist += (t * t) / (x_hist[i] + y_hist[i]);
			k++;
		}
	}

	return dist * (1.f / k);
}

float chi2_patch_distance(const float* channel_histograms[], int nchan, 
	int width, int heigth, 
	int numbins, int patch_size, 
	int ax, int ay, int bx, int by)
{
	float dist = 0;

	for (int i = 0; i < nchan; i++)
	{
		for (int y = -patch_size; y <= patch_size; y++)
		{
			for (int x = -patch_size; x <= patch_size; x++)
			{
				dist += chi2_distance(channel_histograms[i + 1] + ((ay + y) * width + ax + x) * numbins, 
					channel_histograms[i + 1] + ((by + y) * width + bx + x) * numbins, 
					channel_histograms[0][(ay + y) * width + ax + x],
					channel_histograms[0][(by + y) * width + by + y],
					numbins);
			}
		}
	}

	return dist;
}

void rhf_filter_singlescale(int width, int height, int bins, int knn, const float* input_image, const float* channel_histograms[], float* out_image, const RHFParameters& params)
{
	const int num_pixels = width * height;

	memset(out_image, 0, num_pixels * sizeof(float) * 4);

	std::vector<int> counter(num_pixels, 0);

	knn = knn + 1;

	const int window_len = (2 * params.patch_size + 1);
	const int window_size = window_len * window_len;	

	tbb::reader_writer_lock lock;

	tbb::parallel_for(tbb::blocked_range2d<int, int>(0, height, 0, width), [=, &counter, &lock]
	(const tbb::blocked_range2d<int, int>& r)
	{
		std::vector<std::tuple<float, int, int>> nearest_neighours;
		nearest_neighours.reserve(window_size);

		std::vector<float> denoised(window_size * 4, 0);

		for (int py = r.rows().begin(); py < r.rows().end(); py++)
		{
			for (int px = r.cols().begin(); px < r.cols().end(); px++)
			{
				memset(denoised.data(), 0, window_size * 4 * sizeof(float));

				const int window_boundary = std::min(params.patch_size, std::min(width - 1 - px, std::min(height - 1 - py, std::min(px, py))));

				const int min_x = std::max(px - params.search_size, window_boundary);
				const int min_y = std::max(py - params.search_size, window_boundary);
				const int max_x = std::min(px + params.search_size, width - 1 - window_boundary);
				const int max_y = std::min(py + params.search_size, height - 1 - window_boundary);

				float total_weight = 0;
				nearest_neighours.clear();

				for (int sy = min_y; sy <= max_y; sy++)
				{
					for (int sx = min_x; sx <= max_x; sx++)
					{
						if (sx == px && sy == py)
							continue;

						const float dist = chi2_patch_distance(channel_histograms, 3, width, height, bins, window_boundary, px, py, sx, sy);

						nearest_neighours.push_back(std::make_tuple(dist, sx, sy));

						for (int i = 0; i < knn; i++)
						{
							if (dist < std::get<0>(nearest_neighours[i]))
							{
								nearest_neighours[i] = std::make_tuple(dist, sx, sy);
							}
						}
					}
				}

				std::sort(nearest_neighours.begin(), nearest_neighours.end(), [](std::tuple<float, int, int>& lhs, std::tuple<float, int, int>& rhs)
				{
					return std::get<0>(lhs) < std::get<1>(rhs);
				});

				for (int i = 0; i < knn; i++)
				{
					const int sx = std::get<1>(nearest_neighours[i]);
					const int sy = std::get<2>(nearest_neighours[i]);

					for (int patch_y = -window_boundary; patch_y <= window_boundary; patch_y++)
					{
						for (int patch_x = -window_boundary; patch_x <= window_boundary; patch_x++)
						{
							const int target_idx = (params.patch_size + patch_y) * window_len + params.patch_size + patch_x;
							const int src_idx = (sy + patch_y) * width + patch_x + sx;

							denoised[target_idx * 4 + 0] += input_image[src_idx * 4 + 0];
							denoised[target_idx * 4 + 1] += input_image[src_idx * 4 + 1];
							denoised[target_idx * 4 + 2] += input_image[src_idx * 4 + 2];
							denoised[target_idx * 4 + 3] += input_image[src_idx * 4 + 3];
						}
					}

					total_weight += 1.f;
				}

				for (int k = knn; k < nearest_neighours.size(); k++)
				{
					if (std::get<0>(nearest_neighours[k]) < params.threshold)
					{
						const int sx = std::get<1>(nearest_neighours[k]);
						const int sy = std::get<2>(nearest_neighours[k]);

						for (int patch_y = -window_boundary; patch_y <= window_boundary; patch_y++)
						{
							for (int patch_x = -window_boundary; patch_x <= window_boundary; patch_x++)
							{
								const int target_idx = (params.patch_size + patch_y) * window_len + params.patch_size + patch_x;
								const int src_idx = (sy + patch_y) * width + patch_x + sx;

								denoised[target_idx * 4 + 0] += input_image[src_idx * 4 + 0];
								denoised[target_idx * 4 + 1] += input_image[src_idx * 4 + 1];
								denoised[target_idx * 4 + 2] += input_image[src_idx * 4 + 2];
								denoised[target_idx * 4 + 3] += input_image[src_idx * 4 + 3];
							}
						}

						total_weight += 1.f;
					}
				}

				if (total_weight > 0)
				{
					const float weight = 1.0 / total_weight;

					lock.lock();

					for (int patch_y = -window_boundary; patch_y <= window_boundary; patch_y++)
					{
						for (int patch_x = -window_boundary; patch_x <= window_boundary; patch_x++)
						{
							const int src_idx = (params.patch_size + patch_y) * window_len + params.patch_size + patch_x;
							const int target_idx = (py + patch_y) * width + patch_x + px;

							counter[target_idx]++;

							out_image[target_idx * 4 + 0] += denoised[src_idx * 4 + 0] * weight;
							out_image[target_idx * 4 + 1] += denoised[src_idx * 4 + 1] * weight;
							out_image[target_idx * 4 + 2] += denoised[src_idx * 4 + 2] * weight;
							out_image[target_idx * 4 + 3] += denoised[src_idx * 4 + 3] * weight;
						}
					}

					lock.unlock();
				}
			}
		}
	});

	tbb::parallel_for(tbb::blocked_range2d<int, int>(0, height, 0, width), [width, &counter, out_image, input_image](const tbb::blocked_range2d<int, int>& r)
	{
		for (int py = r.rows().begin(); py < r.rows().end(); py++)
		{
			for (int px = r.cols().begin(); px < r.cols().end(); px++)
			{
				const int idx = (py * width + px);

				if (counter[idx] > 0)
				{
					out_image[idx * 4 + 0] /= counter[idx];
					out_image[idx * 4 + 1] /= counter[idx];
					out_image[idx * 4 + 2] /= counter[idx];
					out_image[idx * 4 + 3] /= counter[idx];
				}
				else
				{
					for (int i = 0; i < 4; i++)
					{
						out_image[idx * 4 + i] = input_image[idx * 4 + i];
					}
				}
			}
		}
	});
}

void rhf_filter(int width, int height, int bins, const float* input_image, const float* channel_histograms[], float* out_image, const RHFParameters& params)
{
	const int num_pixels = width * height;

	memset(out_image, 0, num_pixels * 4 * sizeof(float));

	rhf_filter_singlescale(width, height, bins, params.knn, input_image, channel_histograms, out_image, params);
	return;

	double total = 0;

	for (int i = 0; i < width * height; i++)
	{
		total += channel_histograms[0][i];
	}

	OIIO::ImageBuf input_buf(OIIO::ImageSpec(width, height, 4, OIIO::TypeDesc::FLOAT), (void*)input_image);
	OIIO::ImageBuf hist_nsamples(OIIO::ImageSpec(width, height, bins, OIIO::TypeDesc::FLOAT), (void*)channel_histograms[0]);
	OIIO::ImageBuf hist_red(OIIO::ImageSpec(width, height, bins, OIIO::TypeDesc::FLOAT), (void*)channel_histograms[1]);
	OIIO::ImageBuf hist_green(OIIO::ImageSpec(width, height, bins, OIIO::TypeDesc::FLOAT), (void*)channel_histograms[2]);
	OIIO::ImageBuf hist_blue(OIIO::ImageSpec(width, height, bins, OIIO::TypeDesc::FLOAT), (void*)channel_histograms[3]);

	OIIO::ImageBuf old_input_buf;

	/*memcpy(input_buf.localpixels(), 0, width * height * 4 * sizeof(float));
	memcpy(hist_red.localpixels(), 0, width * height * bins * sizeof(float));
	memcpy(hist_green.localpixels(), 0, width * height * bins * sizeof(float));
	memcpy(hist_blue.localpixels(), 0, width * height * bins * sizeof(float));*/

	for (int s = params.nscales - 1; s >= 0; s--)
	{					
		int sw = width;
		int sh = height;		

		OIIO::ImageBuf scaled_input;
		OIIO::ImageBuf scaled_hist_nsamples;
		OIIO::ImageBuf scaled_hist_red;
		OIIO::ImageBuf scaled_hist_green;
		OIIO::ImageBuf scaled_hist_blue;

		if (s > 0)
		{
			const double scale = std::pow(0.5f, s);

			sw *= scale;
			sh *= scale;

			scaled_input.reset(OIIO::ImageSpec(sw, sh, 4, OIIO::TypeDesc::FLOAT));
			scaled_hist_nsamples.reset(OIIO::ImageSpec(sw, sh, 1, OIIO::TypeDesc::FLOAT));
			scaled_hist_red.reset(OIIO::ImageSpec(sw, sh, bins, OIIO::TypeDesc::FLOAT));
			scaled_hist_green.reset(OIIO::ImageSpec(sw, sh, bins, OIIO::TypeDesc::FLOAT));
			scaled_hist_blue.reset(OIIO::ImageSpec(sw, sh, bins, OIIO::TypeDesc::FLOAT));

			OIIO::ImageBufAlgo::resize(scaled_input, input_buf, "gaussian", 2.f);
			OIIO::ImageBufAlgo::resize(scaled_hist_nsamples, hist_nsamples, "gaussian", 2.f);
			OIIO::ImageBufAlgo::resize(scaled_hist_red, hist_red, "gaussian", 2.f);
			OIIO::ImageBufAlgo::resize(scaled_hist_green, hist_green, "gaussian", 2.f);
			OIIO::ImageBufAlgo::resize(scaled_hist_blue, hist_blue, "gaussian", 2.f);

			double total_scaled = 0;

			for (int i = 0; i < sw * sh; i++)
			{
				total_scaled += ((float*)scaled_hist_nsamples.localpixels())[i];
			}

			const float norm = total / total_scaled;

			for (int i = 0; i < sw * sh; i++)
			{
				for (int j = 0; j < bins; j++)
				{
					((float*)scaled_hist_red.localpixels())[i * bins + j] *= norm;
					((float*)scaled_hist_green.localpixels())[i * bins + j] *= norm;
					((float*)scaled_hist_blue.localpixels())[i * bins + j] *= norm;
				}
			}
		}
		else
		{
			scaled_input.swap(input_buf);
			scaled_hist_nsamples.swap(hist_nsamples);
			scaled_hist_red.swap(hist_red);
			scaled_hist_green.swap(hist_green);
			scaled_hist_blue.swap(hist_blue);
		}

		OIIO::ImageBuf out_buf(OIIO::ImageSpec(sw, sh, 4, OIIO::TypeDesc::FLOAT));

		int knn_scaled = s > 0 ? 0 : params.knn;

		const float* scaled_hists[] = 
		{
			(float*)scaled_hist_nsamples.localpixels(),
			(float*)scaled_hist_red.localpixels(),
			(float*)scaled_hist_green.localpixels(),
			(float*)scaled_hist_blue.localpixels()
		};

		rhf_filter_singlescale(sw, sh, bins, knn_scaled, (float*)scaled_input.localpixels(), scaled_hists, (float*)out_buf.localpixels(), params);

		if (s < params.nscales - 1)
		{

		}

		if (s == 0)
		{
			memcpy(out_image, out_buf.localpixels(), width * height * 4 * sizeof(float));
		}

		old_input_buf.swap(scaled_input);
	}		
}