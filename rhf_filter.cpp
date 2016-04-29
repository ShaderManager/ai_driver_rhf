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

	for (int i = 0; i < numbins; i++)
	{
		if (x_hist[i] + y_hist[i] > 0)
		{
			float t = nsy * x_hist[i] - nsx * y_hist[i];

			dist += (t * t) / ((nsx * nsy) * (x_hist[i] + y_hist[i]));
			k++;
		}
	}

	return dist * (1.f / k);
}

template<int nchans = 3> float chi2_patch_distance(const float* channel_histograms[],
	int width, int heigth, 
	int numbins, int patch_size, 
	int ax, int ay, int bx, int by)
{
	float dist = 0;

	
	for (int y = -patch_size; y <= patch_size; y++)
	{
		for (int x = -patch_size; x <= patch_size; x++)
		{
			for (int i = 0; i < nchans; i++)
			{
				dist += chi2_distance(channel_histograms[1 + i] + ((ay + y) * width + ax + x) * numbins,
					channel_histograms[1 + i] + ((by + y) * width + bx + x) * numbins,
					channel_histograms[0][(ay + y) * width + ax + x],
					channel_histograms[0][(by + y) * width + by + y],
					numbins);
			}
		}
	}

	return dist;
}

void rhf_filter_singlescale(int width, int height, int bins, int knn, 
	const float* input_image, const float* channel_histograms[], 
	float* out_image, const RHFParameters& params)
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

						const float dist = chi2_patch_distance(channel_histograms, width, height, bins, window_boundary, px, py, sx, sy);

						nearest_neighours.push_back(std::make_tuple(dist, sx, sy));
					}
				}

				std::sort(nearest_neighours.begin(), nearest_neighours.end(), [](const std::tuple<float, int, int>& lhs, const std::tuple<float, int, int>& rhs)
				{
					return std::get<0>(lhs) < std::get<0>(rhs);
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

void gaussian_downscale(const float* input_image, float* scaled_image, int nchan, 
	int src_width, int src_height, int dest_width, int dest_height, 
	float scale, float sigma_scale)
{
	const float sigma = scale < 1.0f ? sigma_scale * std::sqrt(1.f / (scale * scale) - 1) : sigma_scale;

	/*
	The size of the kernel is selected to guarantee that the
	the first discarded term is at least 10^precision times smaller
	than the central value. For that, h should be larger than x, with
	e^(-x^2/2sigma^2) = 1/10^precision.
	Then, x = sigma * sqrt( 2 * prec * ln(10) ).
	*/
	static const float precision = 2.0;
	const int kernel_radius = std::ceil(sigma * std::sqrt(2.0 * precision * std::log(10.0)));
	const int kernel_size = 2 * kernel_radius + 1;

	std::vector<float> convolved(dest_width * src_height * nchan, 0);

	std::vector<float> kernel(kernel_size, 0);

	const auto gaussian_kernel = [&kernel, kernel_size]
	(float sigma, float mean)
	{
		float sum = 0;

		for (int i = 0; i < kernel_size; i++)
		{
			const float x = ((float)i - mean) / sigma;
			kernel[i] = std::exp(-0.5f * x * x);
			sum += kernel[i];
		}

		if (sum > 0)
		{
			sum = 1.f / sum;

			for (int i = 0; i < kernel_size; i++)
			{
				kernel[i] *= sum;
			}
		}
	};

	// First pass: X axis
	for (int px = 0; px < dest_width; px++)
	{
		const float origin_x = ((float)px + 0.5f) / scale;
		const int origin_center_x = std::floor(origin_x);
		const float mean_x = kernel_radius + origin_x - origin_center_x - 0.5f;

		gaussian_kernel(sigma, mean_x);

		for (int py = 0; py < src_height; py++)
		{
			for (int chan = 0; chan < nchan; chan++)
			{
				float sum = 0;

				for (int i = 0; i < kernel_size; i++)
				{
					int idx = origin_center_x - kernel_radius + i;

					// Wrap x coordinate
					while (idx < 0) idx += 2 * src_width;
					while (idx >= 2 * src_width) idx -= 2 * src_width;
					if (idx >= src_width)
					{
						idx = 2 * src_width - 1 - idx;
					}

					sum += input_image[(py * src_width + idx) * nchan + chan] * kernel[i];
				}

				convolved[(py * dest_width + px) * nchan + chan] = sum;
			}
		}
	}

	// Second pass: Y axis
	for (int py = 0; py < dest_height; py++)
	{
		const float origin_y = ((float)py + 0.5f) / scale;
		const int origin_center_y = std::floor(origin_y);
		const float mean_y = kernel_radius + origin_y - origin_center_y - 0.5f;

		gaussian_kernel(sigma, mean_y);

		for (int px = 0; px < dest_width; px++)
		{
			for (int chan = 0; chan < nchan; chan++)
			{
				float sum = 0;

				for (int i = 0; i < kernel_size; i++)
				{
					int idx = origin_center_y - kernel_radius + i;

					// Wrap y coordinate
					while (idx < 0) idx += 2 * src_height;
					while (idx >= 2 * src_height) idx -= 2 * src_height;
					if (idx >= src_height)
					{
						idx = 2 * src_height - 1 - idx;
					}

					sum += convolved[(idx * dest_width + px) * nchan + chan] * kernel[i];
				}

				scaled_image[(py * dest_width + px) * nchan + chan] = sum;
			}
		}
	}
}

void bicubic_interpolation(const float* input_image, float* out_image, int nchan,
	int src_width, int src_height, int dest_width, int dest_height)
{
	const float scale_x = (float)src_width / dest_width; // 1 / scale_x, scale_x = srcw/destw
	const float scale_y = (float)src_height / dest_height; // 1 / scale_y, scale_y = srch/desth

	std::vector<float> convolved(dest_width * src_height * nchan, 0);

	// Key's function
	const auto cubic = [](float* coeffs, float t, float a)
	{
		const float t2 = t * t;
		const float at = a * t;

		coeffs[0] = a * t2 * (1.f - t);
		coeffs[1] = (2 * a + 3 - (a + 2) * t) * t2 - at;
		coeffs[2] = ((a + 2) * t - a - 3) * t2 + 1;
		coeffs[3] = a * (t - 2) * t2 + at;
	};

	const auto compute_idx = [](int tx, int ty, int w, int h)
	{
		tx = std::min(tx, 2 * w - tx - 1);
		tx = std::max(tx, -tx - 1);

		ty = std::min(ty, 2 * h - ty - 1);
		ty = std::max(ty, -ty - 1);

		return ty * w + tx;
	};

	// First pass
	for (int px = 0; px < dest_width; px++)
	{
		float origin_x = ((float)px + 0.5f) * scale_x;

		if (origin_x < 0 || origin_x > src_width)
		{
			for (int py = 0; py < src_height; py++)
			{
				memset(convolved.data() + (py * dest_width + px) * nchan, 0, sizeof(float) * nchan);
			}
		}
		else
		{
			origin_x -= 0.5f;
			const int center_x = std::floor(origin_x);
			const float frac_x = origin_x - center_x;

			float c[4];
			cubic(c, frac_x, -0.5f);

			if (center_x - 1 >= 0 && center_x + 2 < src_width)
			{
				for (int py = 0; py < src_height; py++)
				{
					for (int chan = 0; chan < nchan; chan++)
					{
						float sum = 0;
						for (int i = -1; i <= 2; i++)
						{
							sum += c[2 - i] * input_image[(py * src_width + center_x + i) * nchan + chan];
						}

						convolved[(py * dest_width + px) * nchan + chan] = sum;
					}
				}
			}
			else
			{
				for (int py = 0; py < src_height; py++)
				{
					for (int chan = 0; chan < nchan; chan++)
					{
						float sum = 0;
						for (int i = -1; i <= 2; i++)
						{
							const int idx = compute_idx(center_x + i, py, src_width, src_height);

							sum += c[2 - i] * input_image[idx * nchan + chan];
						}

						convolved[(py * dest_width + px) * nchan + chan] = sum;
					}
				}
			}
		}
	}

	// Second pass
	for (int py = 0; py < dest_height; py++)
	{
		float origin_y = ((float)py + 0.5f) * scale_y;

		if (origin_y < 0 || origin_y > src_height)
		{
			memset(out_image + py * dest_width * nchan, 0, dest_width * sizeof(float) * nchan);			
		}
		else
		{
			origin_y -= 0.5f;
			const int center_y = std::floor(origin_y);
			const float frac_y = origin_y - center_y;

			float c[4];
			cubic(c, frac_y, -0.5f);

			if (center_y - 1 >= 0 && center_y + 2 < src_height)
			{
				for (int px = 0; px < dest_width; px++)
				{
					for (int chan = 0; chan < nchan; chan++)
					{
						float sum = 0;
						for (int i = -1; i <= 2; i++)
						{
							sum += c[2 - i] * convolved[((center_y + i) * dest_width + px) * nchan + chan];
						}

						out_image[(py * dest_width + px) * nchan + chan] = sum;
					}
				}
			}
			else
			{
				for (int px = 0; px < dest_width; px++)
				{
					for (int chan = 0; chan < nchan; chan++)
					{
						float sum = 0;
						for (int i = -1; i <= 2; i++)
						{
							const int idx = compute_idx(px, center_y + i, dest_width, src_height);

							sum += c[2 - i] * convolved[idx * nchan + chan];
						}

						out_image[(py * dest_width + px) * nchan + chan] = sum;
					}
				}
			}
		}
	}
}

void rhf_filter(int width, int height, int bins, const float* input_image, const float* channel_histograms[], float* out_image, const RHFParameters& params)
{
	const int num_pixels = width * height;

	/*rhf_filter_singlescale(width, height, bins, 0, input_image, channel_histograms, out_image, params);
	return;*/

	double total = 0;

	for (int i = 0; i < width * height; i++)
	{
		total += channel_histograms[0][i];
	}

	std::vector<float> scaled_input_image_buf(width * height * 4, 0);
	std::vector<float> scaled_out_buf(width * height * 4, 0);
	std::vector<float> scaled_old_out_buf(width * height * 4, 0);
	std::vector<float> temp_buf(width * height * 4, 0);
	std::vector<float> temp_buf2(width * height * 4, 0);

	std::vector<float> scaled_hist_nsamples_buf(width * height, 0);
	std::vector<float> scaled_hist_red_buf(width * height * bins, 0);
	std::vector<float> scaled_hist_green_buf(width * height * bins, 0);
	std::vector<float> scaled_hist_blue_buf(width * height * bins, 0);

	static const float sigma_scale = 0.55f;

	int prev_width = width;
	int prev_height = height;

	for (int s = params.nscales - 1; s >= 0; s--)
	{					
		int sw = width;
		int sh = height;

		const float* scaled_input_image = input_image;

		const float* scaled_hists[4] =
		{
			channel_histograms[0],
			channel_histograms[1],
			channel_histograms[2],
			channel_histograms[3],
		};

		if (s > 0)
		{
			const double scale = std::pow(0.5f, s);

			sw = std::floor(sw * scale);
			sh = std::floor(sh * scale);

			gaussian_downscale(input_image, scaled_input_image_buf.data(), 4, width, height, sw, sh, scale, sigma_scale);
			gaussian_downscale(channel_histograms[0], scaled_hist_nsamples_buf.data(), 1, width, height, sw, sh, scale, sigma_scale);
			gaussian_downscale(channel_histograms[1], scaled_hist_red_buf.data(), bins, width, height, sw, sh, scale, sigma_scale);
			gaussian_downscale(channel_histograms[2], scaled_hist_green_buf.data(), bins, width, height, sw, sh, scale, sigma_scale);
			gaussian_downscale(channel_histograms[3], scaled_hist_blue_buf.data(), bins, width, height, sw, sh, scale, sigma_scale);

			double total_scaled = 0;

			for (int i = 0; i < sw * sh; i++)
			{
				total_scaled += scaled_hist_nsamples_buf[i];
			}

			const float norm = total / total_scaled;

			for (int i = 0; i < sw * sh; i++)
			{
				for (int j = 0; j < bins; j++)
				{
					scaled_hist_red_buf[i * bins + j] *= norm;
					scaled_hist_green_buf[i * bins + j] *= norm;
					scaled_hist_blue_buf[i * bins + j] *= norm;
				}
			}

			scaled_input_image = scaled_input_image_buf.data();
			scaled_hists[0] = scaled_hist_nsamples_buf.data();
			scaled_hists[1] = scaled_hist_red_buf.data();
			scaled_hists[2] = scaled_hist_green_buf.data();
			scaled_hists[3] = scaled_hist_blue_buf.data();
		}

		int knn_scaled = s > 0 ? -1 : params.knn;

		rhf_filter_singlescale(sw, sh, bins, knn_scaled, scaled_input_image, scaled_hists, scaled_out_buf.data(), params);

		if (s < params.nscales - 1)
		{	
			int ssw = std::floor(sw * 0.5);
			int ssh = std::floor(sh * 0.5);

			gaussian_downscale(scaled_out_buf.data(), temp_buf.data(), 4, sw, sh, ssw, ssh, 0.5f, sigma_scale);
			bicubic_interpolation(temp_buf.data(), temp_buf2.data(), 4, ssw, ssh, sw, sh);

			for (int i = 0; i < sw * sh; i++)
			{
				scaled_out_buf[i * 4 + 0] -= temp_buf2[i * 4 + 0];
				scaled_out_buf[i * 4 + 1] -= temp_buf2[i * 4 + 1];
				scaled_out_buf[i * 4 + 2] -= temp_buf2[i * 4 + 2];
				scaled_out_buf[i * 4 + 3] -= temp_buf2[i * 4 + 3];
			}

			bicubic_interpolation(scaled_old_out_buf.data(), temp_buf.data(), 4, prev_width, prev_height, sw, sh);

			for (int i = 0; i < sw * sh; i++)
			{
				scaled_out_buf[i * 4 + 0] += temp_buf[i * 4 + 0];
				scaled_out_buf[i * 4 + 1] += temp_buf[i * 4 + 1];
				scaled_out_buf[i * 4 + 2] += temp_buf[i * 4 + 2];
				scaled_out_buf[i * 4 + 3] += temp_buf[i * 4 + 3];
			}
		}

		if (s == 0)
		{
			memcpy(out_image, scaled_out_buf.data(), width * height * 4 * sizeof(float));
		}

		prev_width = sw;
		prev_height = sh;

		std::swap(scaled_out_buf, scaled_old_out_buf);
	}		
}