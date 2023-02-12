

__kernel void blur_kernel(__global uchar* input, __global uchar* output, int width, int height, int inStep, int level) {
	
	int y = get_global_id(0);
	int x = get_global_id(1);
	
	const int in_c = inStep / width;

	int in_loc = y * inStep + (in_c * x);
	int out_loc = in_loc;

	double channel[3];
	int n_pixel = 0;

	for (int blur_row = y - level; blur_row < y + level; blur_row++)
	{
		for (int blur_col = x - level; blur_col < x + level; blur_col++)
		{
			if (blur_row >= 0 && blur_row < height && blur_col >= 0 && blur_col < width)
			{
				in_loc = blur_row * inStep + (in_c * blur_col);
				for (int i = 0; i < in_c; ++i) {
					channel[i] += (double)(input[in_loc + i]);
				}
				n_pixel++; 
			}
		}
	}
	double avg = 0;
	if (n_pixel != 0)
	{
		for (int i = 0; i < in_c; i++)
		{
			avg = (double)(channel[i] / n_pixel);
			//assert(avg <= 255);
			//assert(n_pixel < ((2 * level + 1)* (2 * level + 1)));
			output[out_loc + i] = (uchar)avg;
			channel[i] = 0;
		}
		n_pixel = 0;
	}
}



