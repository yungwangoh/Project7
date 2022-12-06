__kernel void innerProduct(__global float* inputs, __global float* outputs,
__global float* local_sum, __global float* filters, int N) 
{

	// ä�� D1���� input �� �� ä�θ��� �ٸ� filter�� inner product�Ͽ� ä�� D2���� output�� ����

	int output_channel_num = get_global_id(1); // D2 => j  0~31
	int input_channel_num = get_global_id(0); // D1 => i 0~2
	int input_channel_size = get_global_size(0); // D1 == 3
	int local_id = get_local_id(0);

	__global float* input = inputs + N * N * input_channel_num;
	__global float* output = outputs + N * N * output_channel_num;
	__global float* filter = filters + 3 * 3 * (output_channel_num * input_channel_size + input_channel_num);
	float sum = 0.0;

	/* convolution 3x3 */
	for (int i = 0; i < N; i++) { // N * N ���굵 ����ȭ �ʿ�
		for (int j = 0; j < N; j++) {
			sum = 0.0;
			for (int k = 0; k < 3; k++) {
				for (int l = 0; l < 3; l++) {
					int x = i + k - 1;
					int y = j + l - 1;
					if (x >= 0 && x < N && y >= 0 && y < N) 
						sum += input[x * N + y] * filter[k * 3 + l];
					
				}
			}
			local_sum[N * N * local_id + N * i + j] = sum; // ����ȭ ��� �ٸ� ��� ����ϱ�
		}
	}

	//barrier(CLK_LOCAL_MEM_FENCE);

	if (local_id == 0) {
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				for (int k = 0; k < get_local_size(0); k++) {
					output[N * i + j] += local_sum[N * N * k + N * i + j];
				}
			}
		}
	}
	
}