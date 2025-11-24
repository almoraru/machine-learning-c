/******************************************************************************/
/*                                                                            */
/*                       ______                                               */
/*                    .-"      "-.                                            */
/*                   /            \                                           */
/*       _          |              |          _                               */
/*      ( \         |,  .-.  .-.  ,|         / )                              */
/*       > "=._     | )(__/  \__)( |     _.=" <                               */
/*      (_/"=._"=._ |/     /\     \| _.="_.="\_)                              */
/*             "=._ (_     ^^     _)"_.="                                     */
/*                 "=\__|IIIIII|__/="                                         */
/*                _.="| \IIIIII/ |"=._                                        */
/*      _     _.="_.="\          /"=._"=._     _                              */
/*     ( \_.="_.="     `--------`     "=._"=._/ )                             */
/*      > _.="                            "=._ <                              */
/*     (_/                                    \_)                             */
/*                                                                            */
/*     Filename: main.c                                                       */
/*     By: espadara <espadara@pirate.capn.gg>                                 */
/*     Created: 2025/07/25 21:30:41 by espadara                               */
/*      Updated: 2025/11/24 08:49:15 by espadara                              */
/*                                                                            */
/******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>

// --- MATRIX ENGINE ---

typedef struct {
    size_t rows, cols;
    float *es;
} Mat;

#define MAT_AT(m, i, j) (m).es[(i)*(m).cols + (j)]

Mat mat_alloc(size_t rows, size_t cols) {
    Mat m;
    m.rows = rows; m.cols = cols;
    m.es = malloc(sizeof(*m.es) * rows * cols);
    return m;
}

void mat_rand(Mat m, float low, float high) {
    for (size_t i = 0; i < m.rows * m.cols; ++i)
        m.es[i] = (float)rand() / (float)RAND_MAX * (high - low) + low;
}

void mat_dot(Mat dst, Mat a, Mat b) {
    assert(a.cols == b.rows);
    assert(dst.rows == a.rows && dst.cols == b.cols);
    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {
            MAT_AT(dst, i, j) = 0;
            for (size_t k = 0; k < a.cols; ++k)
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
        }
    }
}

void mat_sum(Mat dst, Mat a) {
    assert(dst.rows == a.rows && dst.cols == a.cols);
    for (size_t i = 0; i < dst.rows * dst.cols; ++i)
        dst.es[i] += a.es[i];
}

void mat_sig(Mat m) {
    for (size_t i = 0; i < m.rows * m.cols; ++i)
        m.es[i] = 1.f / (1.f + expf(-m.es[i]));
}

void mat_print(Mat m, const char *name) {
    printf("%s: [\n", name);
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) printf("    %f ", MAT_AT(m, i, j));
        printf("\n");
    }
    printf("]\n");
}

// --- XOR SPECIFIC ---

// 1. The Model Structure
// Instead of named floats, we have named Matrices
typedef struct {
    Mat w1, b1; // Layer 1 (Hidden)
    Mat w2, b2; // Layer 2 (Output)
} Xor;

Xor xor_alloc(void) {
    Xor m;
    m.w1 = mat_alloc(2, 2); m.b1 = mat_alloc(1, 2);
    m.w2 = mat_alloc(2, 1); m.b2 = mat_alloc(1, 1);
    mat_rand(m.w1, 0, 1); mat_rand(m.b1, 0, 1);
    mat_rand(m.w2, 0, 1); mat_rand(m.b2, 0, 1);
    return m;
}

// 2. The Forward Pass
// x (input) -> w1/b1 -> sigmoid -> w2/b2 -> sigmoid -> y (output)
void forward(Xor m, Mat x, Mat y) {
    // We need scratchpads for intermediate calculations
    // In production, allocate these ONCE outside the loop to save speed
    Mat a1 = mat_alloc(1, 2); // Result of x * w1

    // Layer 1
    mat_dot(a1, x, m.w1);
    mat_sum(a1, m.b1);
    mat_sig(a1);

    // Layer 2
    mat_dot(y, a1, m.w2);
    mat_sum(y, m.b2);
    mat_sig(y);

    // Cleanup scratchpad
    free(a1.es);
}

// 3. The Cost Function
float train_data[][3] = {{0,0,0}, {0,1,1}, {1,0,1}, {1,1,0}};

float cost(Xor m) {
    float result = 0.0f;
    Mat x = mat_alloc(1, 2);
    Mat y = mat_alloc(1, 1);

    for (size_t i = 0; i < 4; ++i) {
        MAT_AT(x, 0, 0) = train_data[i][0];
        MAT_AT(x, 0, 1) = train_data[i][1];

        forward(m, x, y);

        float d = MAT_AT(y, 0, 0) - train_data[i][2];
        result += d * d;
    }

    free(x.es); free(y.es);
    return result / 4.0f;
}

// 4. Finite Difference (Wiggling the weights)
// Helper to wiggle one matrix
void finite_diff_mat(Mat m, Mat g, Xor model, float eps) {
    float c = cost(model);
    for (size_t i = 0; i < m.rows * m.cols; ++i) {
        float saved = m.es[i];
        m.es[i] += eps;
        float next_cost = cost(model);
        m.es[i] = saved; // Restore immediately
        g.es[i] = (next_cost - c) / eps;
    }
}

// Apply the gradients
void apply_diff(Mat m, Mat g, float rate) {
    for (size_t i = 0; i < m.rows * m.cols; ++i)
        m.es[i] -= rate * g.es[i];
}

int main(void) {
    srand(time(0));
    Xor m = xor_alloc();
    Xor g = xor_alloc(); // Gradient model to store the "nudges"

    printf("Initial Cost: %f\n", cost(m));

    float eps = 1e-1;
    float rate = 1e-1;

    // TRAINING LOOP
    for (size_t i = 0; i < 50000; ++i) {
        // Calculate gradients for all 4 matrices
        finite_diff_mat(m.w1, g.w1, m, eps);
        finite_diff_mat(m.b1, g.b1, m, eps);
        finite_diff_mat(m.w2, g.w2, m, eps);
        finite_diff_mat(m.b2, g.b2, m, eps);

        // Apply them
        apply_diff(m.w1, g.w1, rate);
        apply_diff(m.b1, g.b1, rate);
        apply_diff(m.w2, g.w2, rate);
        apply_diff(m.b2, g.b2, rate);

        if (i % 5000 == 0) printf("Gen %zu: %f\n", i, cost(m));
    }

    // VERIFICATION
    printf("-----------------\nFinal Results:\n");
    Mat x = mat_alloc(1, 2);
    Mat y = mat_alloc(1, 1);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            MAT_AT(x, 0, 0) = i;
            MAT_AT(x, 0, 1) = j;
            forward(m, x, y);
            printf("%zu ^ %zu = %f\n", i, j, MAT_AT(y, 0, 0));
        }
    }

    return 0;
}
