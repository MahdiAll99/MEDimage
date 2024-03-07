from copy import deepcopy
from typing import Union

import numpy as np
try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.autoinit import context
    from pycuda.compiler import SourceModule
except ImportError:
    print("PyCUDA is not installed. Please install it to use the textural filters.")
    import_failed = True

from ..processing.discretisation import discretize
from .textural_filters_kernels import glcm_kernel, single_glcm_kernel


class TexturalFilter():
    """The Textural filter class. This class is used to apply textural filters to an image. The textural filters are
    chosen from the following families: GLCM, NGTDM, GLDZM, GLSZM, NGLDM, GLRLM. The computation is done using CUDA."""

    def __init__(
                self,
                family: str,
                size: int = 3,
                local: bool = False
                ):

        """
        The constructor for the textural filter class.

        Args:
            family (str): The family of the textural filter.
            size (int, optional): The size of the kernel, which will define the filter kernel dimension.
            local (bool, optional): If true, the discrete will be computed locally, else globally.
        
        Returns:
            None.
        """

        assert size % 2 == 1 and size > 0, "size should be a positive odd number."
        assert isinstance(family, str) and family.upper() in ["GLCM", "NGTDM", "GLDZM", "GLSZM", "NGLDM", "GLRLM"],\
            "family should be a string and should be one of the following: GLCM, NGTDM, GLDZM, GLSZM, NGLDM, GLRLM."

        self.family = family
        self.size = size
        self.local = local
        self.glcm_features = [
            "Fcm_joint_max",
            "Fcm_joint_avg",
            "Fcm_joint_var",
            "Fcm_joint_entr",
            "Fcm_diff_avg",
            "Fcm_diff_var",
            "Fcm_diff_entr",
            "Fcm_sum_avg",
            "Fcm_sum_var",
            "Fcm_sum_entr",
            "Fcm_energy",
            "Fcm_contrast",
            "Fcm_dissimilarity",
            "Fcm_inv_diff",
            "Fcm_inv_diff_norm",
            "Fcm_inv_diff_mom",
            "Fcm_inv_diff_mom_norm",
            "Fcm_inv_var",
            "Fcm_corr",
            "Fcm_auto_corr",
            "Fcm_clust_tend",
            "Fcm_clust_shade",
            "Fcm_clust_prom",
            "Fcm_info_corr1",
            "Fcm_info_corr2"
        ]

    def __glcm_filter(
            self,
            input_images: np.ndarray,
            discretization : dict,
            user_set_min_val: float,
            feature = None
        ) -> np.ndarray:
        """
        Apply a textural filter to the input image.

        Args:
            input_images (ndarray): The images to filter.
            discretization (dict): The discretization parameters.
            user_set_min_val (float): The minimum value to use for the discretization.
            family (str, optional): The family of the textural filter.
            feature (str, optional): The feature to extract from the family. if not specified, all the features of the
                family will be extracted.
        
        Returns:
            ndarray: The filtered image.
        """

        if feature:
            if isinstance(feature, str):
                assert feature in self.glcm_features,\
                    "feature should be a string or an integer and should be one of the following: " + ", ".join(self.glcm_features) + "."
            elif isinstance(feature, int):
                assert feature in range(len(self.glcm_features)),\
                    "feature's index should be an integer between 0 and " + str(len(self.glcm_features) - 1) + "."
            else:
                raise TypeError("feature should be an integer or a string from the following list: " + ", ".join(self.glcm_features) + ".")
                

        # Pre-processing of the input volume
        padding_size = (self.size - 1) // 2
        input_images = np.pad(input_images[:, :, :], padding_size, mode="constant", constant_values=np.nan)
        input_images_copy = deepcopy(input_images)

        # Set up the strides
        strides = (
            input_images_copy.shape[2] * input_images_copy.shape[1] * input_images_copy.dtype.itemsize,
            input_images_copy.shape[2] * input_images_copy.dtype.itemsize,
            input_images_copy.dtype.itemsize
        )
        input_images = np.lib.stride_tricks.as_strided(input_images, shape=input_images.shape, strides=strides)
        input_images[:,:,:] = input_images_copy[:, :, :]

        if self.local:
            # Discretization (to get the global max value)
            if discretization['type'] == "FBS":
                print("Warning: FBS local discretization is equivalent to global discretization.")
                n_q = discretization['bw']
            elif discretization['type'] == "FBN" and discretization['adapted']:
                n_q = (np.nanmax(input_images) - np.nanmin(input_images)) // discretization['bw']
                user_set_min_val = np.nanmin(input_images)
            elif discretization['type'] == "FBN":
                n_q = discretization['bn']
                user_set_min_val = np.nanmin(input_images)
            else:
                raise ValueError("Discretization should be either FBS or FBN.")
            
            temp_vol, _ = discretize(
                vol_re=input_images,
                discr_type=discretization['type'],
                n_q=n_q,
                user_set_min_val=user_set_min_val,
                ivh=False
            )

            # Initialize the filtering parameters
            max_vol = np.nanmax(temp_vol)

            del temp_vol
        
        else:
            # Discretization
            if discretization['type'] == "FBS":
                n_q = discretization['bw']
            elif discretization['type'] == "FBN":
                n_q = discretization['bn']
                user_set_min_val = np.nanmin(input_images)
            else:
                raise ValueError("Discretization should be either FBS or FBN.")
            
            input_images, _ = discretize(
                vol_re=input_images,
                discr_type=discretization['type'],
                n_q=n_q,
                user_set_min_val=user_set_min_val,
                ivh=False
            )

            # Initialize the filtering parameters
            max_vol = np.nanmax(input_images)

        volume_copy = deepcopy(input_images)

        # Filtering
        if feature is not None:
            # Select the feature to compute
            feature = self.glcm_features.index(feature) if isinstance(feature, str) else feature

            # Initialize the kernel
            kernel_glcm = single_glcm_kernel.substitute(
                max_vol=int(max_vol),
                filter_size=self.size,
                shape_volume_0=int(volume_copy.shape[0]),
                shape_volume_1=int(volume_copy.shape[1]),
                shape_volume_2=int(volume_copy.shape[2]),
                discr_type=discretization['type'],
                n_q=n_q,
                min_val=user_set_min_val,
                feature_index=feature
            )

        else:
            # Create the final volume to store the results
            input_images = np.zeros((input_images.shape[0], input_images.shape[1], input_images.shape[2], 25), dtype=np.float32)

            # Fill with nan
            input_images[:] = np.nan

            # Initialize the kernel
            kernel_glcm = glcm_kernel.substitute(
                max_vol=int(max_vol),
                filter_size=self.size,
                shape_volume_0=int(volume_copy.shape[0]),
                shape_volume_1=int(volume_copy.shape[1]),
                shape_volume_2=int(volume_copy.shape[2]),
                discr_type=discretization['type'],
                n_q=n_q,
                min_val=user_set_min_val
            )

        # Compile the CUDA kernel
        if not import_failed:
            mod = SourceModule(kernel_glcm, no_extern_c=True)
            if self.local:
                process_loop_kernel = mod.get_function("glcm_filter_local")
            else:
                process_loop_kernel = mod.get_function("glcm_filter_global")

            # Allocate GPU memory
            volume_gpu = cuda.mem_alloc(input_images.nbytes)
            volume_gpu_copy = cuda.mem_alloc(volume_copy.nbytes)

            # Copy data to the GPU
            cuda.memcpy_htod(volume_gpu, input_images)
            cuda.memcpy_htod(volume_gpu_copy, volume_copy)

            # Set up the grid and block dimensions
            block_dim = (16, 16, 1)  # threads per block
            grid_dim = (
                int((volume_copy.shape[0] - 1) // block_dim[0] + 1),
                int((volume_copy.shape[1] - 1) // block_dim[1] + 1),
                int((volume_copy.shape[2] - 1) // block_dim[2] + 1)
            )   # blocks in the grid

            # Run the kernel
            process_loop_kernel(volume_gpu, volume_gpu_copy, block=block_dim, grid=grid_dim)

            # Synchronize to ensure all CUDA operations are complete
            context.synchronize()

            # Copy data back to the CPU
            cuda.memcpy_dtoh(input_images, volume_gpu)

            # Free the allocated GPU memory
            volume_gpu.free()
            volume_gpu_copy.free()
            del volume_copy

            # unpad the volume
            if feature: # 3D (single-feature)
                input_images = input_images[padding_size:-padding_size, padding_size:-padding_size, padding_size:-padding_size]
            else: # 4D (all features)
                input_images = input_images[padding_size:-padding_size, padding_size:-padding_size, padding_size:-padding_size, :]

            return input_images

        else:
            return None
    
    def __call__(
            self,
            input_images: np.ndarray,
            discretization : dict,
            user_set_min_val: float,
            family: str = "GLCM",
            feature : str = None,
            size: int = None,
            local: bool = False
        ) -> np.ndarray:
        """
        Apply a textural filter to the input image.

        Args:
            input_images (ndarray): The images to filter.
            discretization (dict): The discretization parameters.
            user_set_min_val (float): The minimum value to use for the discretization.
            family (str, optional): The family of the textural filter.
            feature (str, optional): The feature to extract from the family. if not specified, all the features of the
                family will be extracted.
            size (int, optional): The filter size.
            local (bool, optional): If true, the discretization will be computed locally, else globally.
        
        Returns:
            ndarray: The filtered image.
        """
        # Initialization
        if family:
            self.family = family
        if size:
            self.size = size
        if local:
            self.local = local

        # Filtering
        if self.family.lower() == "glcm":
            filtered_images = self.__glcm_filter(input_images, discretization, user_set_min_val, feature)
        else:
            raise NotImplementedError("Only GLCM is implemented for now.")

        return filtered_images
