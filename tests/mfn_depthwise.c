/* Check padding, SIMD tails and stride against direct spatial convolution. */
#include "../src/facex_mfn.c"

int main(void)
{
    enum { channels = 3, max_width = 57, height = 17 };
    float x[channels * height * max_width], y[channels * height * max_width];
    float w[channels * 9], bias[] = {0.17f, -0.25f, 0.33f};
    for (size_t i = 0; i < sizeof(x) / sizeof(x[0]); i++)
        x[i] = ((int)(i % 91) - 54) * 0.0181f;
    for (int i = 0; i < channels * 9; i++)
        w[i] = (i % 7 - 3) * 0.137f;

    for (int width = 1; width <= max_width; width++) {
        for (int stride = 1; stride <= 2; stride++) {
            MfnConv conv = {.w = w, .b = bias, .c_in = channels, .stride = stride};
            conv2d_dw3x3_nchw(x, &conv, height, width, y);
            int ow = (width - 1) / stride + 1, oh = (height - 1) / stride + 1;
            for (int c = 0; c < channels; c++) {
                for (int row = 0; row < oh; row++) {
                    for (int col = 0; col < ow; col++) {
                        double ref = bias[c];
                        for (int tap = 0; tap < 9; tap++) {
                            int iy = row * stride + tap / 3 - 1;
                            int ix = col * stride + tap % 3 - 1;
                            if (iy >= 0 && iy < height && ix >= 0 && ix < width)
                                ref += (double)x[(c * height + iy) * width + ix] * w[c * 9 + tap];
                        }
                        if (!(fabs(y[(c * oh + row) * ow + col] - ref) < 1e-6)) {
                            fprintf(stderr, "Mismatch at width %d stride %d channel %d row %d col %d\n",
                                    width, stride, c, row, col);
                            return 1;
                        }
                    }
                }
            }
        }
    }
    return 0;
}
