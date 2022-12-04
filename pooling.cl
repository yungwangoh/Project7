__kernel void pooling(__global float *input, __global float *output, int N) 
{
    // int i, j, k, l;
	// for (i = 0; i < N; i++) {
	// 	for (j = 0; j < N; j++) {
	// 		float max = 0;
	// 		for (k = 0; k < 2; k++) {
	// 			for (l = 0; l < 2; l++) {
	// 				float pixel = input[(i * 2 + k) * 2 * N + j * 2 + l];
	// 				max = (max > pixel) ? max : pixel;
	// 			}
	// 		}
	// 		output[i * N + j] = max;
	// 	}
	// }
    int i = get_global_id(0);
    int j = get_global_id(1);

    float max = 0;
    for(int k = 0; k < 2; k++) {
        for(int l = 0; l < 2; l++) 
        {
            float pixel = input[(i * 2 + k) * 2 * N + j * 2 + l];
            max = (max > pixel) ? max : pixel;
        }
    }
    output[i * N + j] = max;
}