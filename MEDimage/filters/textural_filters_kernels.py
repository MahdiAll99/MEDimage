from string import Template

glcm_kernel = Template("""
#include <stdio.h>
#include <math.h>
#include <iostream>

# define MAX_SIZE ${max_vol}
# define FILTER_SIZE ${filter_size}

// Function flatten a 3D matrix into a 1D vector
__device__ float * reshape(float(*matrix)[FILTER_SIZE][FILTER_SIZE]) {
    //size of array
    const int size = FILTER_SIZE* FILTER_SIZE* FILTER_SIZE;
    float flattened[size];
    int index = 0;
    for (int i = 0; i < FILTER_SIZE; ++i) {
		for (int j = 0; j < FILTER_SIZE; ++j) {
			for (int k = 0; k < FILTER_SIZE; ++k) {
				flattened[index] = matrix[i][j][k];
                index++;
			}
		}
	}
    return flattened;
}

// Function to perform histogram equalisation of the ROI imaging intensities
__device__ void discretize(float vol_quant_re[FILTER_SIZE][FILTER_SIZE][FILTER_SIZE], 
                            float max_val, float min_val=${min_val}) {
    
    // PARSING ARGUMENTS
    float n_q = ${n_q};
    const char* discr_type = "${discr_type}";

    // DISCRETISATION
    if (discr_type == "FBS") {
        float w_b = n_q;
        for (int i = 0; i < FILTER_SIZE; i++) {
            for (int j = 0; j < FILTER_SIZE; j++) {
                for (int k = 0; k < FILTER_SIZE; k++) {
                    float value = vol_quant_re[i][j][k];
                    if (!isnan(value)) {
                        vol_quant_re[i][j][k] = floorf((value - min_val) / w_b) + 1.0;
                    }
                }
            }
        }
    }
    else if (discr_type == "FBN") {
        float w_b = (max_val - min_val) / n_q;
        for (int i = 0; i < FILTER_SIZE; i++) {
            for (int j = 0; j < FILTER_SIZE; j++) {
                for (int k = 0; k < FILTER_SIZE; k++) {
                    float value = vol_quant_re[i][j][k];
                    if (!isnan(value)) {
                        vol_quant_re[i][j][k] = floorf(n_q * ((value - min_val) / (max_val - min_val))) + 1.0;
                        if (value == max_val) {
							vol_quant_re[i][j][k] = n_q;
						}
                    }
                }
            }
        }
    }
    else {
        printf("ERROR: discretization type not supported");
        assert(false);
    }
}

// Compute the diagonal probability
__device__ float * GLCMDiagProb(float p_ij[MAX_SIZE][MAX_SIZE], float max_vol) {
    int valK[MAX_SIZE];
    for (int i = 0; i < (int)max_vol; ++i) {
        valK[i] = i;
    }
    float p_iminusj[MAX_SIZE] = { 0.0 };
    for (int iterationK = 0; iterationK < (int)max_vol; ++iterationK) {
        int k = valK[iterationK];
        float p = 0.0;
        for (int i = 0; i < (int)max_vol; ++i) {
            for (int j = 0; j < (int)max_vol; ++j) {
                if (k - fabsf(i - j) == 0) {
                    p += p_ij[i][j];
                }
            }
        }

        p_iminusj[iterationK] = p;
    }

    return p_iminusj;
}

// Compute the cross-diagonal probability
__device__ float * GLCMCrossDiagProb(float p_ij[MAX_SIZE][MAX_SIZE], float max_vol) {
    float valK[2 * MAX_SIZE - 1];
    // fill valK with 2, 3, 4, ..., 2*max_vol - 1
    for (int i = 0; i < 2 * (int)max_vol - 1; ++i) {
        valK[i] = i + 2;
    }
    float p_iplusj[2*MAX_SIZE - 1] = { 0.0 };

    for (int iterationK = 0; iterationK < 2*(int)max_vol - 1; ++iterationK) {
        int k = valK[iterationK];
        float p = 0.0;
        for (int i = 0; i < (int)max_vol; ++i) {
            for (int j = 0; j < (int)max_vol; ++j) {
                if (k - (i + j + 2) == 0) {
                    p += p_ij[i][j];
                }
            }
        }

        p_iplusj[iterationK] = p;
    }

    return p_iplusj;
}

__device__ void getGLCMmatrix(
    float (*ROIonly)[FILTER_SIZE][FILTER_SIZE], 
    float GLCMfinal[MAX_SIZE][MAX_SIZE], 
    float max_vol, 
    bool distCorrection = true) 
{
    // PARSING "distCorrection" ARGUMENT
   
    const int Ng = MAX_SIZE;
    float levels[Ng] = {0};
    // initialize levels to 1, 2, 3, ..., 15
    for (int i = 0; i < (int)max_vol; ++i) {
		levels[i] = i + 1;
	}

    float levelTemp = max_vol + 1;

    for (int i = 0; i < FILTER_SIZE; ++i) {
        for (int j = 0; j < FILTER_SIZE; ++j) {
            for (int k = 0; k < FILTER_SIZE; ++k) {
                if (isnan(ROIonly[i][j][k])) {
                    ROIonly[i][j][k] = levelTemp;
                }
            }
        }
    }

    int dim_x = FILTER_SIZE;
    int dim_y = FILTER_SIZE;
    int dim_z = FILTER_SIZE;

    // Reshape the 3D matrix to a 1D vector
    float *q2;
    q2 = reshape(ROIonly);

    // Combine levels and level_temp into qs
    float qs[Ng + 1] = {0};
    for (int i = 0; i < (int)max_vol + 1; ++i) {
        if (i == (int)max_vol) {
			qs[i] = levelTemp;
            break;
		}
        qs[i] = levels[i];
	}
    const int lqs = Ng + 1;

    // Create a q3 matrix and assign values based on qs
    int q3[FILTER_SIZE* FILTER_SIZE* FILTER_SIZE] = {0};

    // fill q3 with 0s
    for (int i = 0; i < FILTER_SIZE * FILTER_SIZE * FILTER_SIZE; ++i) {
        q3[i] = 0;
    }
    for (int k = 0; k < (int)max_vol + 1; ++k) {
        for (int i = 0; i < FILTER_SIZE * FILTER_SIZE * FILTER_SIZE; ++i) {
            if (fabsf(q2[i] - qs[k]) < 1.19209e-07) {
                q3[i] = k;
            }
        }
    }

    // Reshape q3 back to the original dimensions (dimX, dimY, dimZ)
    float reshaped_q3[FILTER_SIZE][FILTER_SIZE][FILTER_SIZE];

    int index = 0;
    for (int i = 0; i < dim_x; ++i) {
        for (int j = 0; j < dim_y; ++j) {
            for (int k = 0; k < dim_z; ++k) {
                reshaped_q3[i][j][k] = q3[index++];
            }
        }
    }


    float GLCM[lqs][lqs] = {0};

    // fill GLCM with 0s
    for (int i = 0; i < (int)max_vol + 1; ++i) {
        for (int j = 0; j < (int)max_vol + 1; ++j) {
			GLCM[i][j] = 0;
		}
	}

    for (int i = 1; i <= dim_x; ++i) {
        int i_min = max(1, i - 1);
        int i_max = min(i + 1, dim_x);
        for (int j = 1; j <= dim_y; ++j) {
            int j_min = max(1, j - 1);
            int j_max = min(j + 1, dim_y);
            for (int k = 1; k <= dim_z; ++k) {
                int k_min = max(1, k - 1);
                int k_max = min(k + 1, dim_z);
                int val_q3 = reshaped_q3[i - 1][j - 1][k - 1];
                for (int I2 = i_min; I2 <= i_max; ++I2) {
                    for (int J2 = j_min; J2 <= j_max; ++J2) {
                        for (int K2 = k_min; K2 <= k_max; ++K2) {
                            if (I2 == i && J2 == j && K2 == k) {
                                continue;
                            }
                            else {
                                int val_neighbor = reshaped_q3[I2 - 1][J2 - 1][K2 - 1];
                                if (distCorrection) {
                                    // Discretization length correction
                                    GLCM[val_q3][val_neighbor] +=
                                        sqrtf(fabsf(I2 - i) +
                                            fabsf(J2 - j) +
                                            fabsf(K2 - k));
                                }
                                else {
                                    GLCM[val_q3][val_neighbor] += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Eliminate last row and column
    for (int i = 0; i < (int)max_vol; ++i) {
		for (int j = 0; j < (int)max_vol; ++j) {
			GLCMfinal[i][j] = GLCM[i][j];
		}
	}
}

__device__ void computeGLCMFeatures(float(*vol)[FILTER_SIZE][FILTER_SIZE], float features[25], float max_vol, bool distCorrection) {

    float GLCM[MAX_SIZE][MAX_SIZE] = { 0.0 };

    // Call function with specified distCorrection
    getGLCMmatrix(vol, GLCM, max_vol, distCorrection);

    // Normalize GLCM
    float sumGLCM = 0.0;
    for (int i = 0; i < (int)max_vol; ++i) {
        for (int j = 0; j < (int)max_vol; ++j) {
            sumGLCM += GLCM[i][j];
        }
    }
    for (int i = 0; i < (int)max_vol; ++i) {
        for (int j = 0; j < (int)max_vol; ++j) {
            GLCM[i][j] /= sumGLCM;
        }
    }

    // Compute textures
    // // Number of gray levels
    const int Ng = MAX_SIZE;
    float vectNg[Ng];

    // fill vectNg with 1, 2, ..., Ng
    for (int i = 0; i < (int)max_vol; ++i) {
        vectNg[i] = i + 1;
    }

    // Create meshgird of size Ng x Ng
    float colGrid[Ng][Ng] = { 0.0 };
    float rowGrid[Ng][Ng] = { 0.0 };

    for (int i = 0; i < (int)max_vol; ++i) {
        for (int j = 0; j < (int)max_vol; ++j) {
            colGrid[i][j] = vectNg[j];
        }
    }
    for (int j = 0; j < int(max_vol); ++j) {
        for (int i = 0; i < int(max_vol); ++i) {
            rowGrid[i][j] = vectNg[i];
        }
    }
    int step_i = 0;
    int step_j = 0;

    // Joint maximum
    float joint_max = 0.0;
    for (int i = 0; i < (int)max_vol; ++i) {
        for (int j = 0; j < (int)max_vol; ++j) {
            joint_max = max(joint_max, GLCM[i][j]);
        }
    }
    features[0] = joint_max;

    // Joint average
    float u = 0.0;
    for (int i = 0; i < (int)max_vol; ++i) {
        step_i = 0;
        for (int j = 0; j < (int)max_vol; ++j) {
            u += GLCM[i][j] * rowGrid[i][j];
            step_i++;
        }
        step_j++;
    }
    features[1] = u;

    // Joint variance
    step_j = 0;
    float var = 0.0;
    u = 0.0;
    for (int i = 0; i < (int)max_vol; ++i) {
        step_i = 0;
        for (int j = 0; j < (int)max_vol; ++j) {
            u += GLCM[i][j] * rowGrid[i][j];
            step_i++;
        }
        step_j++;
    }
    for (int i = 0; i < (int)max_vol; ++i) {
        step_i = 0;
        for (int j = 0; j < (int)max_vol; ++j) {
            var += GLCM[i][j] * powf(rowGrid[i][j] - u, 2);
            step_i++;
        }
        step_j++;
    }
    features[2] = var;

    // Joint entropy
    float entropy = 0.0;
    for (int i = 0; i < (int)max_vol; ++i) {
        for (int j = 0; j < (int)max_vol; ++j) {
            if (GLCM[i][j] > 0.0) {
                entropy += GLCM[i][j] * log2f(GLCM[i][j]);
            }
        }
    }
    features[3] = -entropy;

    // Difference average
    float* p_iminusj;
    p_iminusj = GLCMDiagProb(GLCM, max_vol);
    float diff_avg = 0.0;
    float k[Ng];
    // fill k with 0, 1, ..., Ng - 1
    for (int i = 0; i < int(max_vol); ++i) {
        k[i] = i;
    }
    for (int i = 0; i < int(max_vol); ++i) {
        diff_avg += p_iminusj[i] * k[i];
    }
    features[4] = diff_avg;

    // Difference variance
    diff_avg = 0.0;
    // fill k with 0, 1, ..., Ng - 1
    for (int i = 0; i < int(max_vol); ++i) {
        k[i] = i;
    }
    for (int i = 0; i < int(max_vol); ++i) {
        diff_avg += p_iminusj[i] * k[i];
    }
    float diff_var = 0.0;
    step_i = 0;
    for (int i = 0; i < int(max_vol); ++i) {
        diff_var += p_iminusj[i] * powf(k[i] - diff_avg, 2);
        step_i++;
    }
    features[5] = diff_var;

    // Difference entropy
    float diff_entropy = 0.0;
    for (int i = 0; i < int(max_vol); ++i) {
        if (p_iminusj[i] > 0.0) {
            diff_entropy += p_iminusj[i] * log2f(p_iminusj[i]);
        }
    }
    features[6] = -diff_entropy;

    // Sum average
    float k1[2 * Ng - 1];
    // fill k with 2, 3, ..., 2 * Ng
    for (int i = 0; i < 2 * int(max_vol) - 1; ++i) {
        k1[i] = i + 2;
    }
    float sum_avg = 0.0;
    float* p_iplusj = GLCMCrossDiagProb(GLCM, max_vol);
    for (int i = 0; i < 2 * int(max_vol) - 1; ++i) {
        sum_avg += p_iplusj[i] * k1[i];
    }
    features[7] = sum_avg;

    // Sum variance
    float sum_var = 0.0;
    for (int i = 0; i < 2 * int(max_vol) - 1; ++i) {
        sum_var += p_iplusj[i] * powf(k1[i] - sum_avg, 2);
    }
    features[8] = sum_var;

    // Sum entropy
    float sum_entr = 0.0;
    for (int i = 0; i < 2 * int(max_vol) - 1; ++i) {
        if (p_iplusj[i] > 0.0) {
            sum_entr += p_iplusj[i] * log2f(p_iplusj[i]);
        }
    }
    features[9] = -sum_entr;

    // Angular second moment (energy)
    float energy = 0.0;
    for (int i = 0; i < int(max_vol); ++i) {
        for (int j = 0; j < int(max_vol); ++j) {
            energy += powf(GLCM[i][j], 2);
        }
    }
    features[10] = energy;

    // Contrast
    float contrast = 0.0;
    for (int i = 0; i < int(max_vol); ++i) {
        for (int j = 0; j < int(max_vol); ++j) {
            contrast += powf(rowGrid[i][j] - colGrid[i][j], 2) * GLCM[i][j];
        }
    }
    features[11] = contrast;

    // Dissimilarity
    float dissimilarity = 0.0;
    for (int i = 0; i < int(max_vol); ++i) {
        for (int j = 0; j < int(max_vol); ++j) {
            dissimilarity += fabsf(rowGrid[i][j] - colGrid[i][j]) * GLCM[i][j];
        }
    }
    features[12] = dissimilarity;

    // Inverse difference
    float inv_diff = 0.0;
    for (int i = 0; i < int(max_vol); ++i) {
        for (int j = 0; j < int(max_vol); ++j) {
            inv_diff += GLCM[i][j] / (1 + fabsf(rowGrid[i][j] - colGrid[i][j]));
        }
    }
    features[13] = inv_diff;

    // Inverse difference normalized
    float invDiffNorm = 0.0;
    for (int i = 0; i < int(max_vol); ++i) {
        for (int j = 0; j < int(max_vol); ++j) {
            invDiffNorm += GLCM[i][j] / (1 + fabsf(rowGrid[i][j] - colGrid[i][j]) / int(max_vol));
        }
    }
    features[14] = invDiffNorm;

    // Inverse difference moment
    float invDiffMom = 0.0;
    for (int i = 0; i < int(max_vol); ++i) {
        for (int j = 0; j < int(max_vol); ++j) {
            invDiffMom += GLCM[i][j] / (1 + powf((rowGrid[i][j] - colGrid[i][j]), 2));
        }
    }
    features[15] = invDiffMom;

    // Inverse difference moment normalized
    float invDiffMomNorm = 0.0;
    for (int i = 0; i < int(max_vol); ++i) {
        for (int j = 0; j < int(max_vol); ++j) {
            invDiffMomNorm += GLCM[i][j] / (1 + powf((rowGrid[i][j] - colGrid[i][j]), 2) / powf(int(max_vol), 2));
        }
    }
    features[16] = invDiffMomNorm;

    // Inverse variance
    float invVar = 0.0;
    for (int i = 0; i < int(max_vol); i++) {
        for (int j = i + 1; j < int(max_vol); j++) {
            invVar += GLCM[i][j] / powf((i - j), 2);
        }
    }
    features[17] = 2*invVar;

    // Correlation
    float u_i = 0.0;
    float u_j = 0.0;
    float std_i = 0.0;
    float std_j = 0.0;
    float p_i[Ng] = { 0.0 };
    float p_j[Ng] = { 0.0 };

    // sum over rows
    for (int i = 0; i < int(max_vol); i++) {
        for (int j = 0; j < int(max_vol); j++) {
            p_i[i] += GLCM[i][j];
        }
    }
    // sum over columns
    for (int i = 0; i < int(max_vol); i++) {
        for (int j = 0; j < int(max_vol); j++) {
            p_j[j] += GLCM[i][j];
        }
    }
    for (int i = 0; i < int(max_vol); i++) {
        u_i += vectNg[i] * p_i[i];
        u_j += vectNg[i] * p_j[i];
    }
    for (int i = 0; i < int(max_vol); i++) {
        std_i += powf(vectNg[i] - u_i, 2) * p_i[i];
        std_j += powf(vectNg[i] - u_j, 2) * p_j[i];
    }
    std_i = sqrtf(std_i);
    std_j = sqrtf(std_j);

    float tempSum = 0.0;
    for (int i = 0; i < int(max_vol); i++) {
        for (int j = 0; j < int(max_vol); j++) {
            tempSum += rowGrid[i][j] * colGrid[i][j] * GLCM[i][j];
        }
    }
    float correlation = (1 / (std_i * std_j)) * (-u_i * u_j + tempSum);
    features[18] = correlation;

    // Autocorrelation
    float autoCorr = 0.0;
    for (int i = 0; i < int(max_vol); i++) {
        for (int j = 0; j < int(max_vol); j++) {
            autoCorr += rowGrid[i][j] * colGrid[i][j] * GLCM[i][j];
        }
    }
    features[19] = autoCorr;

    // Cluster tendency
    float clusterTend = 0.0;
    for (int i = 0; i < int(max_vol); i++) {
        for (int j = 0; j < int(max_vol); j++) {
            clusterTend += powf(rowGrid[i][j] + colGrid[i][j] - u_i - u_j, 2) * GLCM[i][j];
        }
    }
    features[20] = clusterTend;

    // Cluster shade
    float clusterShade = 0.0;
    for (int i = 0; i < int(max_vol); i++) {
        for (int j = 0; j < int(max_vol); j++) {
            clusterShade += powf(rowGrid[i][j] + colGrid[i][j] - u_i - u_j, 3) * GLCM[i][j];
        }
    }
    features[21] = clusterShade;

    // Cluster prominence
    float clusterProm = 0.0;
    for (int i = 0; i < int(max_vol); i++) {
        for (int j = 0; j < int(max_vol); j++) {
            clusterProm += powf(rowGrid[i][j] + colGrid[i][j] - u_i - u_j, 4) * GLCM[i][j];
        }
    }
    features[22] = clusterProm;

    // First measure of information correlation
    float HXY = 0.0;
    for (int i = 0; i < int(max_vol); i++) {
        for (int j = 0; j < int(max_vol); j++) {
            if (GLCM[i][j] > 0.0) {
                HXY += GLCM[i][j] * log2f(GLCM[i][j]);
            }
        }
    }
    HXY = -HXY;

    float HX = 0.0;
    for (int i = 0; i < int(max_vol); i++) {
        if (p_i[i] > 0.0) {
            HX += p_i[i] * log2f(p_i[i]);
        }
    }
    HX = -HX;

    // Repeat p_i and p_j Ng times
    float p_i_temp[Ng][Ng];
    float p_j_temp[Ng][Ng];
    float p_temp[Ng][Ng];

    for (int i = 0; i < int(max_vol); i++) {
        for (int j = 0; j < int(max_vol); j++) {
            p_i_temp[i][j] = p_i[i];
            p_j_temp[i][j] = p_j[j];
            p_temp[i][j] = p_i_temp[i][j] * p_j_temp[i][j];
        }
    }

    float HXY1 = 0.0;
    for (int i = 0; i < int(max_vol); i++) {
        for (int j = 0; j < int(max_vol); j++) {
            if (p_temp[i][j] > 0.0) {
                HXY1 += GLCM[i][j] * log2f(p_temp[i][j]);
            }
        }
    }
    HXY1 = -HXY1;
    features[23] = (HXY - HXY1) / HX;

    // Second measure of information correlation
    float HXY2 = 0.0;
    for (int i = 0; i < int(max_vol); i++) {
        for (int j = 0; j < int(max_vol); j++) {
            if (p_temp[i][j] > 0.0) {
                HXY2 += p_temp[i][j] * log2f(p_temp[i][j]);
            }
        }
    }
    HXY2 = -HXY2;
    if (HXY > HXY2) {
        features[24] = 0.0;
    }
    else {
        features[24] = sqrtf(1 - expf(-2 * (HXY2 - HXY)));
    }
}

extern "C"
__global__ void glcm_filter_global(
    float vol[${shape_volume_0}][${shape_volume_1}][${shape_volume_2}][25],
    float vol_copy[${shape_volume_0}][${shape_volume_1}][${shape_volume_2}],
    bool distCorrection = false)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < ${shape_volume_0} && j < ${shape_volume_1} && k < ${shape_volume_2} && i >= 0 && j >= 0 && k >= 0) {
        // pad size
        const int padd_size = (FILTER_SIZE - 1) / 2;

        // size vol
        const int size_x = ${shape_volume_0};
        const int size_y = ${shape_volume_1};
        const int size_z = ${shape_volume_2};

        // skip all calculations if vol at position i,j,k is nan
        if (!isnan(vol_copy[i][j][k])) {
            // get submatrix
            float sub_matrix[FILTER_SIZE][FILTER_SIZE][FILTER_SIZE] = {NAN};
            for (int idx_i = 0; idx_i < FILTER_SIZE; ++idx_i) {
                for (int idx_j = 0; idx_j < FILTER_SIZE; ++idx_j) {
                    for (int idx_k = 0; idx_k < FILTER_SIZE; ++idx_k) {
                    if ((i - padd_size + idx_i) >= 0 && (i - padd_size + idx_i) < size_x && 
                        (j - padd_size + idx_j) >= 0 && (j - padd_size + idx_j) < size_y &&
                        (k - padd_size + idx_k) >= 0 && (k - padd_size + idx_k) < size_z) {
                            sub_matrix[idx_i][idx_j][idx_k] = vol_copy[i - padd_size + idx_i][j - padd_size + idx_j][k - padd_size + idx_k];
                        }
                    }
                }
            }

            // get the maximum value of the submatrix
            float max_vol = -3.40282e+38;
            for (int idx_i = 0; idx_i < FILTER_SIZE; ++idx_i) {
                for (int idx_j = 0; idx_j < FILTER_SIZE; ++idx_j) {
                    for (int idx_k = 0; idx_k < FILTER_SIZE; ++idx_k) {
                        max_vol = max(max_vol, sub_matrix[idx_i][idx_j][idx_k]);
                    }
                }
            }
            
            // compute GLCM features
            float features[25] = { 0.0 };
            computeGLCMFeatures(sub_matrix, features, max_vol, false);
            
            // Copy GLCM feature to voxels of the volume
            if (i < size_x && j < size_y && k < size_z){
                for (int idx = 0; idx < 25; ++idx) {
                    vol[i][j][k][idx] = features[idx];
                }
            }
        }
    }
}

extern "C"
__global__ void glcm_filter_local(
    float vol[${shape_volume_0}][${shape_volume_1}][${shape_volume_2}][25],
    float vol_copy[${shape_volume_0}][${shape_volume_1}][${shape_volume_2}],
    bool distCorrection = false)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < ${shape_volume_0} && j < ${shape_volume_1} && k < ${shape_volume_2} && i >= 0 && j >= 0 && k >= 0) {
        // pad size
        const int padd_size = (FILTER_SIZE - 1) / 2;

        // size vol
        const int size_x = ${shape_volume_0};
        const int size_y = ${shape_volume_1};
        const int size_z = ${shape_volume_2};

        // skip all calculations if vol at position i,j,k is nan
        if (!isnan(vol_copy[i][j][k])) {
            // get submatrix
            float sub_matrix[FILTER_SIZE][FILTER_SIZE][FILTER_SIZE] = {NAN};
            for (int idx_i = 0; idx_i < FILTER_SIZE; ++idx_i) {
                for (int idx_j = 0; idx_j < FILTER_SIZE; ++idx_j) {
                    for (int idx_k = 0; idx_k < FILTER_SIZE; ++idx_k) {
                    if ((i - padd_size + idx_i) >= 0 && (i - padd_size + idx_i) < size_x && 
                        (j - padd_size + idx_j) >= 0 && (j - padd_size + idx_j) < size_y &&
                        (k - padd_size + idx_k) >= 0 && (k - padd_size + idx_k) < size_z) {
                            sub_matrix[idx_i][idx_j][idx_k] = vol_copy[i - padd_size + idx_i][j - padd_size + idx_j][k - padd_size + idx_k];
                        }
                    }
                }
            }

            // get the maximum value of the submatrix
            float max_vol = -3.40282e+38;
            for (int idx_i = 0; idx_i < FILTER_SIZE; ++idx_i) {
                for (int idx_j = 0; idx_j < FILTER_SIZE; ++idx_j) {
                    for (int idx_k = 0; idx_k < FILTER_SIZE; ++idx_k) {
                        max_vol = max(max_vol, sub_matrix[idx_i][idx_j][idx_k]);
                    }
                }
            }
            // get the minimum value of the submatrix if discr_type is FBN
            float min_val = 3.40282e+38;
            if ("${discr_type}" == "FBN") {
                for (int idx_i = 0; idx_i < FILTER_SIZE; ++idx_i) {
                    for (int idx_j = 0; idx_j < FILTER_SIZE; ++idx_j) {
                        for (int idx_k = 0; idx_k < FILTER_SIZE; ++idx_k) {
                            min_val = min(min_val, sub_matrix[idx_i][idx_j][idx_k]);
                        }
                    }
                }
                discretize(sub_matrix, max_vol, min_val);
            }

            // If FBS discretize the submatrix with user set minimum value
            else{
                discretize(sub_matrix, max_vol);
            }
                                                          
            // get the maximum value of the submatrix after discretization
            max_vol = -3.40282e+38;
            for (int idx_i = 0; idx_i < FILTER_SIZE; ++idx_i) {
                for (int idx_j = 0; idx_j < FILTER_SIZE; ++idx_j) {
                    for (int idx_k = 0; idx_k < FILTER_SIZE; ++idx_k) {
                        max_vol = max(max_vol, sub_matrix[idx_i][idx_j][idx_k]);
                    }
                }
            }

            // compute GLCM features
            float features[25] = { 0.0 };
            computeGLCMFeatures(sub_matrix, features, max_vol, false);
            
            // Copy GLCM feature to voxels of the volume
            if (i < size_x && j < size_y && k < size_z){
                for (int idx = 0; idx < 25; ++idx) {
                    vol[i][j][k][idx] = features[idx];
                }
            }
        }
    }
}
""")
                       
# Signle-feature kernel
single_glcm_kernel = Template("""
#include <stdio.h>
#include <math.h>
#include <iostream>

# define MAX_SIZE ${max_vol}
# define FILTER_SIZE ${filter_size}

// Function flatten a 3D matrix into a 1D vector
__device__ float * reshape(float(*matrix)[FILTER_SIZE][FILTER_SIZE]) {
    //size of array
    const int size = FILTER_SIZE* FILTER_SIZE* FILTER_SIZE;
    float flattened[size];
    int index = 0;
    for (int i = 0; i < FILTER_SIZE; ++i) {
		for (int j = 0; j < FILTER_SIZE; ++j) {
			for (int k = 0; k < FILTER_SIZE; ++k) {
				flattened[index] = matrix[i][j][k];
                index++;
			}
		}
	}
    return flattened;
}

// Function to perform discretization on the ROI imaging intensities
__device__ void discretize(float vol_quant_re[FILTER_SIZE][FILTER_SIZE][FILTER_SIZE], 
                            float max_val, float min_val=${min_val}) {
    
    // PARSING ARGUMENTS
    float n_q = ${n_q};
    const char* discr_type = "${discr_type}";

    // DISCRETISATION
    if (discr_type == "FBS") {
        float w_b = n_q;
        for (int i = 0; i < FILTER_SIZE; i++) {
            for (int j = 0; j < FILTER_SIZE; j++) {
                for (int k = 0; k < FILTER_SIZE; k++) {
                    float value = vol_quant_re[i][j][k];
                    if (!isnan(value)) {
                        vol_quant_re[i][j][k] = floorf((value - min_val) / w_b) + 1.0;
                    }
                }
            }
        }
    }
    else if (discr_type == "FBN") {
        float w_b = (max_val - min_val) / n_q;
        for (int i = 0; i < FILTER_SIZE; i++) {
            for (int j = 0; j < FILTER_SIZE; j++) {
                for (int k = 0; k < FILTER_SIZE; k++) {
                    float value = vol_quant_re[i][j][k];
                    if (!isnan(value)) {
                        vol_quant_re[i][j][k] = floorf(n_q * ((value - min_val) / (max_val - min_val))) + 1.0;
                        if (value == max_val) {
							vol_quant_re[i][j][k] = n_q;
						}
                    }
                }
            }
        }
    }
    else {
        printf("ERROR: discretization type not supported");
        assert(false);
    }
}

// Compute the diagonal probability
__device__ float * GLCMDiagProb(float p_ij[MAX_SIZE][MAX_SIZE], float max_vol) {
    int valK[MAX_SIZE];
    for (int i = 0; i < (int)max_vol; ++i) {
        valK[i] = i;
    }
    float p_iminusj[MAX_SIZE] = { 0.0 };
    for (int iterationK = 0; iterationK < (int)max_vol; ++iterationK) {
        int k = valK[iterationK];
        float p = 0.0;
        for (int i = 0; i < (int)max_vol; ++i) {
            for (int j = 0; j < (int)max_vol; ++j) {
                if (k - fabsf(i - j) == 0) {
                    p += p_ij[i][j];
                }
            }
        }

        p_iminusj[iterationK] = p;
    }

    return p_iminusj;
}

// Compute the cross-diagonal probability
__device__ float * GLCMCrossDiagProb(float p_ij[MAX_SIZE][MAX_SIZE], float max_vol) {
    float valK[2 * MAX_SIZE - 1];
    // fill valK with 2, 3, 4, ..., 2*max_vol - 1
    for (int i = 0; i < 2 * (int)max_vol - 1; ++i) {
        valK[i] = i + 2;
    }
    float p_iplusj[2*MAX_SIZE - 1] = { 0.0 };

    for (int iterationK = 0; iterationK < 2*(int)max_vol - 1; ++iterationK) {
        int k = valK[iterationK];
        float p = 0.0;
        for (int i = 0; i < (int)max_vol; ++i) {
            for (int j = 0; j < (int)max_vol; ++j) {
                if (k - (i + j + 2) == 0) {
                    p += p_ij[i][j];
                }
            }
        }

        p_iplusj[iterationK] = p;
    }

    return p_iplusj;
}

__device__ void getGLCMmatrix(
    float (*ROIonly)[FILTER_SIZE][FILTER_SIZE], 
    float GLCMfinal[MAX_SIZE][MAX_SIZE], 
    float max_vol, 
    bool distCorrection = true) 
{
    // PARSING "distCorrection" ARGUMENT
   
    const int Ng = MAX_SIZE;
    float levels[Ng] = {0};
    // initialize levels to 1, 2, 3, ..., 15
    for (int i = 0; i < (int)max_vol; ++i) {
		levels[i] = i + 1;
	}

    float levelTemp = max_vol + 1;

    for (int i = 0; i < FILTER_SIZE; ++i) {
        for (int j = 0; j < FILTER_SIZE; ++j) {
            for (int k = 0; k < FILTER_SIZE; ++k) {
                if (isnan(ROIonly[i][j][k])) {
                    ROIonly[i][j][k] = levelTemp;
                }
            }
        }
    }

    int dim_x = FILTER_SIZE;
    int dim_y = FILTER_SIZE;
    int dim_z = FILTER_SIZE;

    // Reshape the 3D matrix to a 1D vector
    float *q2;
    q2 = reshape(ROIonly);

    // Combine levels and level_temp into qs
    float qs[Ng + 1] = {0};
    for (int i = 0; i < (int)max_vol + 1; ++i) {
        if (i == (int)max_vol) {
			qs[i] = levelTemp;
            break;
		}
        qs[i] = levels[i];
	}
    const int lqs = Ng + 1;

    // Create a q3 matrix and assign values based on qs
    int q3[FILTER_SIZE* FILTER_SIZE* FILTER_SIZE] = {0};

    // fill q3 with 0s
    for (int i = 0; i < FILTER_SIZE * FILTER_SIZE * FILTER_SIZE; ++i) {
        q3[i] = 0;
    }
    for (int k = 0; k < (int)max_vol + 1; ++k) {
        for (int i = 0; i < FILTER_SIZE * FILTER_SIZE * FILTER_SIZE; ++i) {
            if (fabsf(q2[i] - qs[k]) < 1.19209e-07) {
                q3[i] = k;
            }
        }
    }

    // Reshape q3 back to the original dimensions (dimX, dimY, dimZ)
    float reshaped_q3[FILTER_SIZE][FILTER_SIZE][FILTER_SIZE];

    int index = 0;
    for (int i = 0; i < dim_x; ++i) {
        for (int j = 0; j < dim_y; ++j) {
            for (int k = 0; k < dim_z; ++k) {
                reshaped_q3[i][j][k] = q3[index++];
            }
        }
    }


    float GLCM[lqs][lqs] = {0};

    // fill GLCM with 0s
    for (int i = 0; i < (int)max_vol + 1; ++i) {
        for (int j = 0; j < (int)max_vol + 1; ++j) {
			GLCM[i][j] = 0;
		}
	}

    for (int i = 1; i <= dim_x; ++i) {
        int i_min = max(1, i - 1);
        int i_max = min(i + 1, dim_x);
        for (int j = 1; j <= dim_y; ++j) {
            int j_min = max(1, j - 1);
            int j_max = min(j + 1, dim_y);
            for (int k = 1; k <= dim_z; ++k) {
                int k_min = max(1, k - 1);
                int k_max = min(k + 1, dim_z);
                int val_q3 = reshaped_q3[i - 1][j - 1][k - 1];
                for (int I2 = i_min; I2 <= i_max; ++I2) {
                    for (int J2 = j_min; J2 <= j_max; ++J2) {
                        for (int K2 = k_min; K2 <= k_max; ++K2) {
                            if (I2 == i && J2 == j && K2 == k) {
                                continue;
                            }
                            else {
                                int val_neighbor = reshaped_q3[I2 - 1][J2 - 1][K2 - 1];
                                if (distCorrection) {
                                    // Discretization length correction
                                    GLCM[val_q3][val_neighbor] +=
                                        sqrtf(fabsf(I2 - i) +
                                            fabsf(J2 - j) +
                                            fabsf(K2 - k));
                                }
                                else {
                                    GLCM[val_q3][val_neighbor] += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Eliminate last row and column
    for (int i = 0; i < (int)max_vol; ++i) {
		for (int j = 0; j < (int)max_vol; ++j) {
			GLCMfinal[i][j] = GLCM[i][j];
		}
	}
}

__device__ float computeGLCMFeatures(float(*vol)[FILTER_SIZE][FILTER_SIZE], int feature, float max_vol, bool distCorrection) {

    float GLCM[MAX_SIZE][MAX_SIZE] = { 0.0 };

    // Call function with specified distCorrection
    getGLCMmatrix(vol, GLCM, max_vol, distCorrection);

    // Normalize GLCM
    float sumGLCM = 0.0;
    for (int i = 0; i < (int)max_vol; ++i) {
        for (int j = 0; j < (int)max_vol; ++j) {
			sumGLCM += GLCM[i][j];
		}
	}
    for (int i = 0; i < (int)max_vol; ++i) {
        for (int j = 0; j < (int)max_vol; ++j) {
            GLCM[i][j] /= sumGLCM;
        }
    }

    // Compute textures
    // // Number of gray levels
    const int Ng = MAX_SIZE;
    float vectNg[Ng];

    // fill vectNg with 1, 2, ..., Ng
    for (int i = 0; i < (int)max_vol; ++i) {
		vectNg[i] = i + 1;
	}

    // Create meshgird of size Ng x Ng
    float colGrid[Ng][Ng] = { 0.0 };
    float rowGrid[Ng][Ng] = { 0.0 };

    for (int i = 0; i < (int)max_vol; ++i) {
        for (int j = 0; j < (int)max_vol; ++j) {
            colGrid[i][j] = vectNg[j];
        }
    }
    for (int j = 0; j < int(max_vol); ++j) {
        for (int i = 0; i < int(max_vol); ++i) {
            rowGrid[i][j] = vectNg[i];
        }
    }
    int step_i = 0;
    int step_j = 0;

    // Joint maximum
    if (feature == 0) {
        float joint_max = NAN; 
        for (int i = 0; i < (int)max_vol; ++i) {
            for (int j = 0; j < (int)max_vol; ++j) {
                joint_max = max(joint_max, GLCM[i][j]);
            }
        }
        return joint_max;
    }
    // Joint average
    else if (feature == 1) {
        float u = 0.0;
        for (int i = 0; i < (int)max_vol; ++i) {
            step_i = 0;
            for (int j = 0; j < (int)max_vol; ++j) {
                u += GLCM[i][j] * rowGrid[i][j];
                step_i++;
            }
            step_j++;
        }
        return u;
    }
    // Joint variance
    else if (feature == 2) {
        step_j = 0;
        float var = 0.0;
        float u = 0.0;
        for (int i = 0; i < (int)max_vol; ++i) {
            step_i = 0;
            for (int j = 0; j < (int)max_vol; ++j) {
                u += GLCM[i][j] * rowGrid[i][j];
                step_i++;
            }
            step_j++;
        }
        for (int i = 0; i < (int)max_vol; ++i) {
            step_i = 0;
            for (int j = 0; j < (int)max_vol; ++j) {
                var += GLCM[i][j] * powf(rowGrid[i][j] - u, 2);
                step_i++;
            }
            step_j++;
        }
        return var;
    }
    // Joint entropy
    else if (feature == 3) {
		float entropy = 0.0;
        for (int i = 0; i < (int)max_vol; ++i) {
            for (int j = 0; j < (int)max_vol; ++j) {
                if (GLCM[i][j] > 0.0) {
                    entropy += GLCM[i][j] * log2f(GLCM[i][j]);
				}
			}
		}
		return -entropy;
	}
    // Difference average
    else if (feature == 4) {
        float *p_iminusj;
        p_iminusj = GLCMDiagProb(GLCM, max_vol);
        float diff_avg = 0.0;
        float k[Ng];
        // fill k with 0, 1, ..., Ng - 1
        for (int i = 0; i < int(max_vol); ++i) {
			k[i] = i;
		}
        for (int i = 0; i < int(max_vol); ++i) {
            diff_avg += p_iminusj[i] * k[i];
        }
        return diff_avg;
	}
    // Difference variance
    else if (feature == 5) {
        float* p_iminusj;
        p_iminusj = GLCMDiagProb(GLCM, max_vol);
        float diff_avg = 0.0;
        float k[Ng];
        // fill k with 0, 1, ..., Ng - 1
        for (int i = 0; i < int(max_vol); ++i) {
            k[i] = i;
        }
        for (int i = 0; i < int(max_vol); ++i) {
            diff_avg += p_iminusj[i] * k[i];
        }
        float diff_var = 0.0;
        step_i = 0;
        for (int i = 0; i < int(max_vol); ++i) {
            diff_var += p_iminusj[i] * powf(k[i] - diff_avg, 2);
            step_i++;
        }
        return diff_var;
    }
    // Difference entropy
    else if (feature == 6) {
        float* p_iminusj = GLCMDiagProb(GLCM, max_vol);
		float diff_entropy = 0.0;
        for (int i = 0; i < int(max_vol); ++i) {
            if (p_iminusj[i] > 0.0) {
				diff_entropy += p_iminusj[i] * log2f(p_iminusj[i]);
			}
		}
		return -diff_entropy;
	}
    // Sum average
    else if (feature == 7) {
        float k[2 * Ng - 1];
        // fill k with 2, 3, ..., 2 * Ng
        for (int i = 0; i < 2 * int(max_vol) - 1; ++i) {
			k[i] = i + 2;
		}
        float sum_avg = 0.0;
        float* p_iplusj = GLCMCrossDiagProb(GLCM, max_vol);
        for (int i = 0; i < 2*int(max_vol) - 1; ++i) {
            sum_avg += p_iplusj[i] * k[i];
        }
        return sum_avg;
	}
    // Sum variance
    else if (feature == 8) {
		float k[2 * Ng - 1];
		// fill k with 2, 3, ..., 2 * Ng
        for (int i = 0; i < 2 * int(max_vol) - 1; ++i) {
			k[i] = i + 2;
		}
		float sum_avg = 0.0;
		float* p_iplusj = GLCMCrossDiagProb(GLCM, max_vol);
        for (int i = 0; i < 2 * int(max_vol) - 1; ++i) {
			sum_avg += p_iplusj[i] * k[i];
		}
		float sum_var = 0.0;
        for (int i = 0; i < 2 * int(max_vol) - 1; ++i) {
			sum_var += p_iplusj[i] * powf(k[i] - sum_avg, 2);
		}
		return sum_var;
	}
    // Sum entropy
    else if (feature == 9) {
        float sum_entr = 0.0;
        float* p_iplusj = GLCMCrossDiagProb(GLCM, max_vol);
        for (int i = 0; i < 2 * int(max_vol) - 1; ++i) {
            if (p_iplusj[i] > 0.0) {
                sum_entr += p_iplusj[i] * log2f(p_iplusj[i]);
            }
        }
        return -sum_entr;
    }
    // Angular second moment (energy)
    else if (feature == 10) {
		float energy = 0.0;
        for (int i = 0; i < int(max_vol); ++i) {
            for (int j = 0; j < int(max_vol); ++j) {
                energy += powf(GLCM[i][j], 2);
			}
		}
		return energy;
	}
    // Contrast
    else if (feature == 11) {
		float contrast = 0.0;
        for (int i = 0; i < int(max_vol); ++i) {
            for (int j = 0; j < int(max_vol); ++j) {
                contrast += powf(rowGrid[i][j] - colGrid[i][j], 2) * GLCM[i][j];
			}
		}
		return contrast;
	}
    // Dissimilarity
    else if (feature == 12) {
        float dissimilarity = 0.0;
        for (int i = 0; i < int(max_vol); ++i) {
            for (int j = 0; j < int(max_vol); ++j) {
                dissimilarity += fabsf(rowGrid[i][j] - colGrid[i][j]) * GLCM[i][j];
            }
        }
        return dissimilarity;
    }
    // Inverse difference
    else if (feature == 13) {
		float inv_diff = 0.0;
        for (int i = 0; i < int(max_vol); ++i) {
            for (int j = 0; j < int(max_vol); ++j) {
                inv_diff += GLCM[i][j] / (1 + fabsf(rowGrid[i][j] - colGrid[i][j]));
			}
		}
		return inv_diff;
	}
    // Inverse difference normalized
    else if (feature == 14) {
        float invDiffNorm = 0.0;
        for (int i = 0; i < int(max_vol); ++i) {
            for (int j = 0; j < int(max_vol); ++j) {
                invDiffNorm += GLCM[i][j] / (1 + fabsf(rowGrid[i][j] - colGrid[i][j]) / int(max_vol));
            }
        }
        return invDiffNorm;
    }
    // Inverse difference moment
    else if (feature == 15) {
        float invDiffMom = 0.0;
        for (int i = 0; i < int(max_vol); ++i) {
            for (int j = 0; j < int(max_vol); ++j) {
                invDiffMom += GLCM[i][j] / (1 + powf((rowGrid[i][j] - colGrid[i][j]), 2));
            }
        }
        return invDiffMom;
	}
    // Inverse difference moment normalized
    else if (feature == 16) {
		float invDiffMomNorm = 0.0;
        for (int i = 0; i < int(max_vol); ++i) {
            for (int j = 0; j < int(max_vol); ++j) {
                invDiffMomNorm += GLCM[i][j] / (1 + powf((rowGrid[i][j] - colGrid[i][j]), 2) / powf(int(max_vol), 2));
			}
		}
		return invDiffMomNorm;
	}
    // Inverse variance
    else if (feature == 17) {
        float invVar = 0.0;
        for (int i = 0; i < int(max_vol); i++) {
            for (int j = i + 1; j < int(max_vol); j++) {
                invVar += GLCM[i][j] / powf((i - j), 2);
            }
        }
        return 2*invVar;
    }
    // Correlation
    else if (feature == 18) {
        float u_i = 0.0;
        float u_j = 0.0;
        float std_i = 0.0;
        float std_j = 0.0;
        float p_i[Ng] = { 0.0 };
        float p_j[Ng] = { 0.0 };

        // sum over rows
        for (int i = 0; i < int(max_vol); i++) {
            for (int j = 0; j < int(max_vol); j++) {
                p_i[i] += GLCM[i][j];
            }
        }
        // sum over columns
        for (int i = 0; i < int(max_vol); i++) {
            for (int j = 0; j < int(max_vol); j++) {
                p_j[j] += GLCM[i][j];
            }
        }
        for (int i = 0; i < int(max_vol); i++) {
            u_i += vectNg[i] * p_i[i];
            u_j += vectNg[i] * p_j[i];
        }
        for (int i = 0; i < int(max_vol); i++) {
            std_i += powf(vectNg[i] - u_i, 2) * p_i[i];
            std_j += powf(vectNg[i] - u_j, 2) * p_j[i];
        }
        std_i = sqrtf(std_i);
        std_j = sqrtf(std_j);

        float tempSum = 0.0;
        for (int i = 0; i < int(max_vol); i++) {
            for (int j = 0; j < int(max_vol); j++) {
                tempSum += rowGrid[i][j] * colGrid[i][j] * GLCM[i][j];
            }
        }
        float correlation = (1 / (std_i * std_j)) * (-u_i * u_j + tempSum);
        return correlation;
    }
    // Autocorrelation
    else if (feature == 19) {
        float u_i = 0.0;
        float u_j = 0.0;
        float std_i = 0.0;
        float std_j = 0.0;
        float p_i[Ng] = { 0.0 };
        float p_j[Ng] = { 0.0 };

        // sum over rows
        for (int i = 0; i < int(max_vol); i++) {
            for (int j = 0; j < int(max_vol); j++) {
                p_i[i] += GLCM[i][j];
            }
        }
        // sum over columns
        for (int i = 0; i < int(max_vol); i++) {
            for (int j = 0; j < int(max_vol); j++) {
                p_j[j] += GLCM[i][j];
            }
        }
        for (int i = 0; i < int(max_vol); i++) {
            u_i += vectNg[i] * p_i[i];
            u_j += vectNg[i] * p_j[i];
        }
        for (int i = 0; i < int(max_vol); i++) {
            std_i += powf(vectNg[i] - u_i, 2) * p_i[i];
            std_j += powf(vectNg[i] - u_j, 2) * p_j[i];
        }
        std_i = sqrtf(std_i);
        std_j = sqrtf(std_j);

        float autoCorr = 0.0;
        for (int i = 0; i < int(max_vol); i++) {
            for (int j = 0; j < int(max_vol); j++) {
                autoCorr += rowGrid[i][j] * colGrid[i][j] * GLCM[i][j];
            }
        }
        
        return autoCorr;
    }
    // Cluster tendency
    else if (feature == 20) {
        float u_i = 0.0;
        float u_j = 0.0;
        float p_i[Ng] = { 0.0 };
        float p_j[Ng] = { 0.0 };

        // sum over rows
        for (int i = 0; i < int(max_vol); i++) {
            for (int j = 0; j < int(max_vol); j++) {
                p_i[i] += GLCM[i][j];
            }
        }
        // sum over columns
        for (int i = 0; i < int(max_vol); i++) {
            for (int j = 0; j < int(max_vol); j++) {
                p_j[j] += GLCM[i][j];
            }
        }
        for (int i = 0; i < int(max_vol); i++) {
            u_i += vectNg[i] * p_i[i];
            u_j += vectNg[i] * p_j[i];
        }
        float clusterTend = 0.0;
        for (int i = 0; i < int(max_vol); i++) {
            for (int j = 0; j < int(max_vol); j++) {
                clusterTend += powf(rowGrid[i][j] + colGrid[i][j] - u_i - u_j, 2) * GLCM[i][j];
            }
        }
		return clusterTend;
	}
    // Cluster shade
    else if (feature == 21) {
		float u_i = 0.0;
		float u_j = 0.0;
		float p_i[Ng] = { 0.0 };
		float p_j[Ng] = { 0.0 };

		// sum over rows
        for (int i = 0; i < int(max_vol); i++) {
            for (int j = 0; j < int(max_vol); j++) {
				p_i[i] += GLCM[i][j];
			}
		}
		// sum over columns
        for (int i = 0; i < int(max_vol); i++) {
            for (int j = 0; j < int(max_vol); j++) {
				p_j[j] += GLCM[i][j];
			}
		}
        for (int i = 0; i < int(max_vol); i++) {
			u_i += vectNg[i] * p_i[i];
			u_j += vectNg[i] * p_j[i];
		}
		float clusterShade = 0.0;
        for (int i = 0; i < int(max_vol); i++) {
            for (int j = 0; j < int(max_vol); j++) {
				clusterShade += powf(rowGrid[i][j] + colGrid[i][j] - u_i - u_j, 3) * GLCM[i][j];
			}
		}
		return clusterShade;
	}
    // Cluster prominence
    else if (feature == 22) {
        float u_i = 0.0;
        float u_j = 0.0;
        float p_i[Ng] = { 0.0 };
        float p_j[Ng] = { 0.0 };

        // sum over rows
        for (int i = 0; i < int(max_vol); i++) {
            for (int j = 0; j < int(max_vol); j++) {
                p_i[i] += GLCM[i][j];
            }
        }
        // sum over columns
        for (int i = 0; i < int(max_vol); i++) {
            for (int j = 0; j < int(max_vol); j++) {
                p_j[j] += GLCM[i][j];
            }
        }
        for (int i = 0; i < int(max_vol); i++) {
            u_i += vectNg[i] * p_i[i];
            u_j += vectNg[i] * p_j[i];
        }
        float clusterProm = 0.0;
        for (int i = 0; i < int(max_vol); i++) {
            for (int j = 0; j < int(max_vol); j++) {
                clusterProm += powf(rowGrid[i][j] + colGrid[i][j] - u_i - u_j, 4) * GLCM[i][j];
            }
        }
        return clusterProm;
    }
    // First measure of information correlation
    else if (feature == 23) {
        float p_i[Ng] = { 0.0 };
        float p_j[Ng] = { 0.0 };
        // sum over rows
        for (int i = 0; i < int(max_vol); i++) {
            for (int j = 0; j < int(max_vol); j++) {
                p_i[i] += GLCM[i][j];
            }
        }
        // sum over columns
        for (int i = 0; i < int(max_vol); i++) {
            for (int j = 0; j < int(max_vol); j++) {
                p_j[j] += GLCM[i][j];
            }
        }

        float HXY = 0.0;
        for (int i = 0; i < int(max_vol); i++) {
            for (int j = 0; j < int(max_vol); j++) {
                if (GLCM[i][j] > 0.0) {
                    HXY += GLCM[i][j] * log2f(GLCM[i][j]);
                }
            }
        }
        HXY = -HXY;

        float HX = 0.0;
        for (int i = 0; i < int(max_vol); i++) {
            if (p_i[i] > 0.0) {
                HX += p_i[i] * log2f(p_i[i]);
            }
        }
        HX = -HX;

        // Repeat p_i and p_j Ng times
        float p_i_temp[Ng][Ng];
        float p_j_temp[Ng][Ng];
        float p_temp[Ng][Ng];

        for (int i = 0; i < int(max_vol); i++) {
            for (int j = 0; j < int(max_vol); j++) {
                p_i_temp[i][j] = p_i[i];
                p_j_temp[i][j] = p_j[j];
                p_temp[i][j] = p_i_temp[i][j] * p_j_temp[i][j];
            }
        }

        float HXY1 = 0.0;
        for (int i = 0; i < int(max_vol); i++) {
            for (int j = 0; j < int(max_vol); j++) {
                if (p_temp[i][j] > 0.0) {
                    HXY1 += GLCM[i][j] * log2f(p_temp[i][j]);
                }
            }
        }
        HXY1 = -HXY1;

        return (HXY - HXY1) / HX;
    }
    // Second measure of information correlation
    else if (feature == 24) {
        float p_i[Ng] = { 0.0 };
        float p_j[Ng] = { 0.0 };
        // sum over rows
        for (int i = 0; i < int(max_vol); i++) {
            for (int j = 0; j < int(max_vol); j++) {
                p_i[i] += GLCM[i][j];
            }
        }
        // sum over columns
        for (int i = 0; i < int(max_vol); i++) {
            for (int j = 0; j < int(max_vol); j++) {
                p_j[j] += GLCM[i][j];
            }
        }

        float HXY = 0.0;
        for (int i = 0; i < int(max_vol); i++) {
            for (int j = 0; j < int(max_vol); j++) {
                if (GLCM[i][j] > 0.0) {
                    HXY += GLCM[i][j] * log2f(GLCM[i][j]);
                }
            }
        }
        HXY = -HXY;

        // Repeat p_i and p_j Ng times
        float p_i_temp[Ng][Ng];
        float p_j_temp[Ng][Ng];
        float p_temp[Ng][Ng];

        for (int i = 0; i < int(max_vol); i++) {
            for (int j = 0; j < int(max_vol); j++) {
                p_i_temp[i][j] = p_i[i];
                p_j_temp[i][j] = p_j[j];
                p_temp[i][j] = p_i_temp[i][j] * p_j_temp[i][j];
            }
        }

        float HXY2 = 0.0;
        for (int i = 0; i < int(max_vol); i++) {
            for (int j = 0; j < int(max_vol); j++) {
                if (p_temp[i][j] > 0.0) {
                    HXY2 += p_temp[i][j] * log2f(p_temp[i][j]);
                }
            }
        }
        HXY2 = -HXY2;
        if (HXY > HXY2) {
            return 0.0;
        }
        else {
            return sqrtf(1 - expf(-2 * (HXY2 - HXY)));
        }
    }
    else {
        // Print error message
        printf("Error: feature %d not implemented\\n", feature);
        assert(false);
    }
}

extern "C"
__global__ void glcm_filter_global(
    float vol[${shape_volume_0}][${shape_volume_1}][${shape_volume_2}],
    float vol_copy[${shape_volume_0}][${shape_volume_1}][${shape_volume_2}],
    bool distCorrection = false)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < ${shape_volume_0} && j < ${shape_volume_1} && k < ${shape_volume_2} && i >= 0 && j >= 0 && k >= 0) {
        // pad size
        const int padd_size = (FILTER_SIZE - 1) / 2;

        // size vol
        const int size_x = ${shape_volume_0};
        const int size_y = ${shape_volume_1};
        const int size_z = ${shape_volume_2};

        // skip all calculations if vol at position i,j,k is nan
        if (!isnan(vol_copy[i][j][k])) {
            // get submatrix
            float sub_matrix[FILTER_SIZE][FILTER_SIZE][FILTER_SIZE] = {NAN};
            for (int idx_i = 0; idx_i < FILTER_SIZE; ++idx_i) {
                for (int idx_j = 0; idx_j < FILTER_SIZE; ++idx_j) {
                    for (int idx_k = 0; idx_k < FILTER_SIZE; ++idx_k) {
                    if ((i - padd_size + idx_i) >= 0 && (i - padd_size + idx_i) < size_x && 
                        (j - padd_size + idx_j) >= 0 && (j - padd_size + idx_j) < size_y &&
                        (k - padd_size + idx_k) >= 0 && (k - padd_size + idx_k) < size_z) {
                            sub_matrix[idx_i][idx_j][idx_k] = vol_copy[i - padd_size + idx_i][j - padd_size + idx_j][k - padd_size + idx_k];
                        }
                    }
                }
            }

            // get the maximum value of the submatrix
            float max_vol = -3.40282e+38;
            for (int idx_i = 0; idx_i < FILTER_SIZE; ++idx_i) {
                for (int idx_j = 0; idx_j < FILTER_SIZE; ++idx_j) {
                    for (int idx_k = 0; idx_k < FILTER_SIZE; ++idx_k) {
                        max_vol = max(max_vol, sub_matrix[idx_i][idx_j][idx_k]);
                    }
                }
            }

            // get feature index
            const int feature = ${feature_index};
            
            // compute GLCM features
            float glcm_feature = computeGLCMFeatures(sub_matrix, feature, max_vol, false);
            
            // Copy GLCM feature to voxels of the volume
            if (i < size_x && j < size_y && k < size_z){
                vol[i][j][k] = glcm_feature;
            }
        }
    }
}

extern "C"
__global__ void glcm_filter_local(
    float vol[${shape_volume_0}][${shape_volume_1}][${shape_volume_2}],
    float vol_copy[${shape_volume_0}][${shape_volume_1}][${shape_volume_2}],
    bool distCorrection = false)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < ${shape_volume_0} && j < ${shape_volume_1} && k < ${shape_volume_2} && i >= 0 && j >= 0 && k >= 0) {
        // pad size
        const int padd_size = (FILTER_SIZE - 1) / 2;

        // size vol
        const int size_x = ${shape_volume_0};
        const int size_y = ${shape_volume_1};
        const int size_z = ${shape_volume_2};

        // skip all calculations if vol at position i,j,k is nan
        if (!isnan(vol_copy[i][j][k])) {
            // get submatrix
            float sub_matrix[FILTER_SIZE][FILTER_SIZE][FILTER_SIZE] = {NAN};
            for (int idx_i = 0; idx_i < FILTER_SIZE; ++idx_i) {
                for (int idx_j = 0; idx_j < FILTER_SIZE; ++idx_j) {
                    for (int idx_k = 0; idx_k < FILTER_SIZE; ++idx_k) {
                    if ((i - padd_size + idx_i) >= 0 && (i - padd_size + idx_i) < size_x && 
                        (j - padd_size + idx_j) >= 0 && (j - padd_size + idx_j) < size_y &&
                        (k - padd_size + idx_k) >= 0 && (k - padd_size + idx_k) < size_z) {
                            sub_matrix[idx_i][idx_j][idx_k] = vol_copy[i - padd_size + idx_i][j - padd_size + idx_j][k - padd_size + idx_k];
                        }
                    }
                }
            }

            // get the maximum value of the submatrix
            float max_vol = -3.40282e+38;
            for (int idx_i = 0; idx_i < FILTER_SIZE; ++idx_i) {
                for (int idx_j = 0; idx_j < FILTER_SIZE; ++idx_j) {
                    for (int idx_k = 0; idx_k < FILTER_SIZE; ++idx_k) {
                        max_vol = max(max_vol, sub_matrix[idx_i][idx_j][idx_k]);
                    }
                }
            }
            // get the minimum value of the submatrix if discr_type is FBN
            float min_val = 3.40282e+38;
            if ("${discr_type}" == "FBN") {
                for (int idx_i = 0; idx_i < FILTER_SIZE; ++idx_i) {
                    for (int idx_j = 0; idx_j < FILTER_SIZE; ++idx_j) {
                        for (int idx_k = 0; idx_k < FILTER_SIZE; ++idx_k) {
                            min_val = min(min_val, sub_matrix[idx_i][idx_j][idx_k]);
                        }
                    }
                }
                discretize(sub_matrix, max_vol, min_val);
            }

            // If FBS discretize the submatrix with user set minimum value
            else{
                discretize(sub_matrix, max_vol);
            }
                                                          
            // get the maximum value of the submatrix after discretization
            max_vol = -3.40282e+38;
            for (int idx_i = 0; idx_i < FILTER_SIZE; ++idx_i) {
                for (int idx_j = 0; idx_j < FILTER_SIZE; ++idx_j) {
                    for (int idx_k = 0; idx_k < FILTER_SIZE; ++idx_k) {
                        max_vol = max(max_vol, sub_matrix[idx_i][idx_j][idx_k]);
                    }
                }
            }

            // get feature index
            const int feature = ${feature_index};
            
            // compute GLCM features
            float glcm_feature = computeGLCMFeatures(sub_matrix, feature, max_vol, false);
            
            // Copy GLCM feature to voxels of the volume
            if (i < size_x && j < size_y && k < size_z){
                vol[i][j][k] = glcm_feature;
            }
        }
    }
}
""")