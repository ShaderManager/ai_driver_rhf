#include <ai.h>

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <cassert>

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/filter.h>

#include "rhf_filter.hpp"

AI_DRIVER_NODE_EXPORT_METHODS(rhf_driver_mtds);

struct RHFDriver
{	
	static const uint32_t num_bins = 20;

	float* pixels;
	uint32_t width, height;

	float* nsamples;
	float* hist_red;
	float* hist_green;
	float* hist_blue;

	OIIO::Filter2D* filter;
};

node_parameters
{
	AiParameterSTR("filename", "output.tif");
	AiMetaDataSetBool(mds, "filename", "maya.hide", true);

	AiMetaDataSetStr(mds, NULL, "maya.attr_prefix", "");
	AiMetaDataSetBool(mds, NULL, "display_driver", true);

	// Pixel filter settings
	AiParameterSTR("filter", "gaussian");
	AiParameterFlt("filter_width", 2);

	// RHF params
	AiParameterFLT("threshold", 1.0f);
	AiParameterINT("patch_size", 1);
	AiParameterINT("search_window_size", 6);
	AiParameterINT("scales", 2);
	AiParameterINT("knn", 2);
}

node_initialize
{
	RHFDriver* self = (RHFDriver*)AiMalloc(sizeof(RHFDriver));

	static const char* required_aovs[] = { "RGBA RGBA", NULL };

	AiRawDriverInitialize(node, required_aovs, false, self);
}

node_update
{

}

node_finish
{
	RHFDriver* self = static_cast<RHFDriver*>(AiDriverGetLocalData(node));

	OIIO::Filter2D::destroy(self->filter);

	AiFree(self->pixels);

	AiFree(self);
	AiDriverDestroy(node);
}

driver_supports_pixel_type
{
	return true;
}

driver_extension
{
	return NULL;
}

driver_open
{
	RHFDriver* self = static_cast<RHFDriver*>(AiDriverGetLocalData(node));

	self->width = display_window.maxx - display_window.minx + 1;
	self->height = display_window.maxy - display_window.miny + 1;	

	const float filter_width = AiNodeGetFlt(node, "filter_width");
	self->filter = OIIO::Filter2D::create(AiNodeGetStr(node, "filter"), filter_width, filter_width);

	const int num_pixels = self->width * self->height;
	const int num_channels = 4;

	self->pixels = (float*)AiMalloc(num_pixels * sizeof(float) * num_channels);

	self->nsamples = (float*)AiMalloc(num_pixels * sizeof(float));
	self->hist_red = (float*)AiMalloc(num_pixels * RHFDriver::num_bins * sizeof(float));
	self->hist_green = (float*)AiMalloc(num_pixels * RHFDriver::num_bins * sizeof(float));
	self->hist_blue = (float*)AiMalloc(num_pixels * RHFDriver::num_bins * sizeof(float));

	memset(self->nsamples, 0, num_pixels * sizeof(float));
	memset(self->hist_red, 0, num_pixels * RHFDriver::num_bins* sizeof(float));
	memset(self->hist_green, 0, num_pixels * RHFDriver::num_bins * sizeof(float));
	memset(self->hist_blue, 0, num_pixels * RHFDriver::num_bins * sizeof(float));
}

driver_prepare_bucket
{
}

void update_bin(float sample, float* hist_bins, float gamma = 2.2f, float M = 7.5f, float s = 2.f)
{
	sample = std::max<float>(sample, 0);
	sample = std::pow(sample, 1.f / gamma) / M;
	sample = std::min<float>(sample, s);

	float fbin = sample * (RHFDriver::num_bins - 2);
	const int ibin_low = std::floor(fbin);

	float weight_bin_high, weight_bin_low;
	int index_bin_high, index_bin_low;

	if (ibin_low < RHFDriver::num_bins - 2)
	{
		weight_bin_high = fbin - ibin_low;
		index_bin_low = ibin_low;
		weight_bin_low = 1.f - weight_bin_high;
		index_bin_high = ibin_low + 1;
	}
	else
	{
		weight_bin_high = (sample - 1.f) / (s - 1.f);
		index_bin_low = RHFDriver::num_bins - 2;
		weight_bin_low = 1.f - weight_bin_high;
		index_bin_high = RHFDriver::num_bins - 1;
	}

	hist_bins[index_bin_low] += weight_bin_low;
	hist_bins[index_bin_high] += weight_bin_high;
}

driver_process_bucket
{
	RHFDriver* self = static_cast<RHFDriver*>(AiDriverGetLocalData(node));

	for (int py = bucket_yo; py < bucket_yo + bucket_size_y; py++)
	{
		for (int px = bucket_xo; px < bucket_xo + bucket_size_x; px++)
		{
			AiAOVSampleIteratorInitPixel(sample_iterator, px, py);

			AtRGBA pixel = { 0, 0, 0, 0 };

			float total_weight = 0;

			while (AiAOVSampleIteratorGetNext(sample_iterator))
			{
				const AtPoint2 position = AiAOVSampleIteratorGetOffset(sample_iterator);
				const AtRGBA sample = AiAOVSampleIteratorGetRGBA(sample_iterator);

				const float weight = (*self->filter)(position.x, position.y);

				pixel += weight * sample;
				total_weight += weight;

				self->nsamples[py * self->width + px] += weight;

				update_bin(weight * sample.r, self->hist_red + (py * self->width + px) * RHFDriver::num_bins);
				update_bin(weight * sample.g, self->hist_green + (py * self->width + px) * RHFDriver::num_bins);
				update_bin(weight * sample.b, self->hist_blue + (py * self->width + px) * RHFDriver::num_bins);
			}

			if (total_weight > 0)
			{
				*(AtRGBA*)(self->pixels + (py * self->width + px) * 4) = pixel / total_weight;
			}
		}
	}
}

driver_needs_bucket
{
	return true;
}

driver_write_bucket
{
	//RHFDriver* self = static_cast<RHFDriver*>(AiDriverGetLocalData(node));
}

driver_close
{
	RHFDriver* self = static_cast<RHFDriver*>(AiDriverGetLocalData(node));

	const char* output_name = AiNodeGetStr(node, "filename");

	auto image_out = OIIO::ImageOutput::create(output_name);

	if (image_out)
	{
		if (image_out->open(output_name, OIIO::ImageSpec(self->width, self->height, 4, OIIO::TypeDesc::FLOAT)))
		{
			const float* hists[] = { self->nsamples, self->hist_red, self->hist_green, self->hist_blue };

			float* filtered_image = new float[self->width * self->height * 4];

			RHFParameters params;

			params.patch_size = AiNodeGetInt(node, "patch_size");
			params.search_size = AiNodeGetInt(node, "search_window_size");
			params.nscales = AiNodeGetInt(node, "scales");
			params.knn = AiNodeGetInt(node, "knn");
			params.threshold = AiNodeGetFlt(node, "threshold");

			AiMsgInfo("Starting filtering image with RHF");

			rhf_filter(self->width, self->height, RHFDriver::num_bins, self->pixels, hists, filtered_image, params);

			AiMsgInfo("Filtered! Writing image");

			if (!image_out->write_image(OIIO::TypeDesc::FLOAT, filtered_image))
			{
				AiMsgError("Could not write image: %s", image_out->geterror().c_str());
			}
			image_out->close();

			delete[] filtered_image;
		}
		else
		{
			AiMsgError("Could not open '%s' on write. Reason: %s", output_name, image_out->geterror().c_str());
		}
	}
	else
	{
		AiMsgError("Could not initialize image writer. Reason: %s", OIIO::geterror().c_str());
	}

	delete image_out;
}

node_loader
{
	sprintf(node->version, AI_VERSION);

	switch (i)
	{
	case 0:
		{
			node->methods = rhf_driver_mtds;
			node->output_type = AI_TYPE_RGBA;
			node->name = "driver_rhf";
			node->node_type = AI_NODE_DRIVER;
		}
		break;
	default:
		return false;
	}

	return true;
}
