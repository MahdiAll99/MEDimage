import logging
import os
import warnings
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image
from pydicom.dataset import FileDataset

from .utils.imref import imref3d

_logger = logging.getLogger(__name__)

class MEDimage(object):
    """Organizes all scan data (patientID, imaging data, scan type...). 
    Args:
        MEDimg (MEDimage, optional): A MEDimage instance.
    Attributes:
        patientID (str): Patient ID.
        type (str): Scan type (MRscan, CTscan...).
        format (str): Scan file format. Either 'npy' or 'nifti'.
        dicomH (pydicom.dataset.FileDataset): DICOM header.
        scan (MEDimage.scan): Instance of MEDimage.scan inner class.
    """

    def __init__(self, MEDimg=None) -> None:
        try:
            self.patientID = MEDimg.patientID
        except:
            self.patientID = ""
        try:
            self.type = MEDimg.type
        except:
            self.type = ""
        try:
            self.format = MEDimg.format
        except:
            self.format = ""
        try:
            self.dicomH = MEDimg.dicomH
        except:
            self.dicomH = []
        try:
            self.scan = MEDimg.scan
        except:
            self.scan = self.scan()
    
    @property
    def patientID(self) -> str:
        """
        Patient ID
        Returns:
            patientID (str): Patient ID
        """
        return self._patientID
    
    @patientID.setter
    def patientID(self, patientID: str) -> None:
        """
        Patient ID setter
        Args:
            patientID (str): Patient ID
        """
        self._patientID = patientID
    
    @property
    def type(self) -> str:
        """
        Imaging scan type
        Returns:
            type (str): Imaging scan type
        """
        return self._type
    
    @type.setter
    def type(self, type: str) -> None:
        """
        Imaging scan type
        Args:
            type (str): Imaging scan type
        """
        self._type = type
    
    @property
    def dicomH(self) -> List[FileDataset]:
        """
        DICOM header
        Returns:
            dicomH (List): DICOM header
        """
        return self._dicomH
    
    @dicomH.setter
    def dicomH(self, dicomH: List[FileDataset]) -> None:
        """
        DICOM header
        Args:
            dicomH (List): DICOM header
        """
        self._dicomH = dicomH

    def init_from_nifti(self, NiftiImagePath) -> None:
        """Initializes the MEDimage class using a NIFTI file.
        Args:
            NiftiImagePath (Path): NIFTI file path.
        Returns:
            None.
        
        """
        self.patientID = os.path.basename(NiftiImagePath).split("_")[0]
        self.type = os.path.basename(NiftiImagePath).split(".")[-3]
        self.format = "nifti"
        self.scan.set_orientation(orientation="Axial")
        self.scan.set_patientPosition(patientPosition="HFS")
        self.scan.ROI.get_ROI_from_path(ROI_path=os.path.dirname(NiftiImagePath), 
                                        ID=Path(NiftiImagePath).name.split("(")[0])
        self.scan.volume.data = nib.load(NiftiImagePath).get_fdata()
        # RAS to LPS
        self.scan.volume.convert_to_LPS()
        self.scan.volume.scanRot = None


    class scan:
        """Organizes all imaging data (volume and ROI). 
        Args:
            orientation (str, optional): Imaging data orientation (axial, sagittal or coronal).
            patientPosition (str, optional): Patient position specifies the position of the 
                patient relative to the imaging equipment space (HFS, HFP...).
        Attributes:
            volume (object): Instance of MEDimage.scan.volume inner class.
            ROI (object): Instance of MEDimage.scan.ROI inner class.
            orientation (str): Imaging data orientation (axial, sagittal or coronal).
            patientPosition (str): Patient position specifies the position of the 
                patient relative to the imaging equipment space (HFS, HFP...).
        """
        def __init__(self, orientation=None, patientPosition=None):
            self.volume = self.volume() 
            self.ROI = self.ROI()
            self.orientation = orientation
            self.patientPosition = patientPosition

        def set_patientPosition(self, patientPosition):
            self.patientPosition = patientPosition

        def set_orientation(self, orientation):
            self.orientation = orientation
        
        def set_volume(self, volume):
            self.volume = volume
        
        def set_ROI(self, *args):
            self.ROI = self.ROI(args)

        def get_ROI_from_indexes(self, key):
            """
            Extract ROI data using the saved indexes (Indexes of 1's).
            Args:
                ket (int): ROI index (A volume can have multiple ROIs).
            Returns:
                ndarray: n-dimensional array of ROI data.
            
            """
            roi_volume = np.zeros_like(self.volume.data).flatten()
            roi_volume[self.ROI.get_indexes(key)] = 1
            return roi_volume.reshape(self.volume.data.shape)

        def get_indexes_by_ROIname(self, ROIname : str):
            """
            Extract ROI data using ROI name..
            Args:
                ROIname (str): String of the ROI name (A volume can have multiple ROIs).
            Returns:
                ndarray: n-dimensional array of ROI data.
            
            """
            ROIname_key = list(self.ROI.roi_names.values()).index(ROIname)
            roi_volume = np.zeros_like(self.volume.data).flatten()
            roi_volume[self.ROI.get_indexes(ROIname_key)] = 1
            return roi_volume.reshape(self.volume.data.shape)

        def display(self, _slice: int = None) -> None:
            """Displays slices from imaging data with the ROI contour in XY-Plane.
            Args:
                _slice (int, optional): Index of the slice you want to plot.
            Returns:
                None.
            
            """
            # extract slices containing ROI
            size_m = self.volume.data.shape
            i = np.arange(0, size_m[0])
            j = np.arange(0, size_m[1])
            k = np.arange(0, size_m[2])
            ind_mask = np.nonzero(self.get_ROI_from_indexes(0))
            J, I, K = np.meshgrid(j, i, k, indexing='ij')
            I = I[ind_mask]
            J = J[ind_mask]
            K = K[ind_mask]
            slices = np.unique(K)

            vol_data = self.volume.data.swapaxes(0, 1)[:, :, slices]
            roi_data = self.get_ROI_from_indexes(0).swapaxes(0, 1)[:, :, slices]        
            
            rows = int(np.round(np.sqrt(len(slices))))
            columns = int(np.ceil(len(slices) / rows))
            
            plt.set_cmap(plt.gray())
            
            # plot only one slice
            if _slice:
                fig, ax =  plt.subplots(1, 1, figsize=(10, 5))
                ax.axis('off')
                ax.set_title(_slice)
                ax.imshow(vol_data[:, :, _slice])
                im = Image.fromarray((roi_data[:, :, _slice]))
                ax.contour(im, colors='red', linewidths=0.4, alpha=0.45)
                lps_ax = fig.add_subplot(1, columns, 1)
            
            # plot multiple slices containing an ROI.
            else:
                fig, axs =  plt.subplots(rows, columns+1, figsize=(20, 10))
                s = 0
                for i in range(0,rows):
                    for j in range(0,columns):
                        axs[i,j].axis('off')
                        if s < len(slices):
                            axs[i,j].set_title(str(s))
                            axs[i,j].imshow(vol_data[:, :, s])
                            im = Image.fromarray((roi_data[:, :, s]))
                            axs[i,j].contour(im, colors='red', linewidths=0.4, alpha=0.45)
                        s += 1
                    axs[i,columns].axis('off')
                lps_ax = fig.add_subplot(1, columns+1, axs.shape[1])

            fig.suptitle('XY-Plane')
            fig.tight_layout()
            
            # add the coordinates system
            lps_ax.axis([-1.5, 1.5, -1.5, 1.5])
            lps_ax.set_title("Coordinates system")
            
            lps_ax.quiver([-0.5], [0], [1.5], [0], scale_units='xy', angles='xy', scale=1.0, color='green')
            lps_ax.quiver([-0.5], [0], [0], [-1.5], scale_units='xy', angles='xy', scale=3, color='blue')
            lps_ax.quiver([-0.5], [0], [1.5], [1.5], scale_units='xy', angles='xy', scale=3, color='red')
            lps_ax.text(1.0, 0, "L")
            lps_ax.text(-0.3, -0.5, "P")
            lps_ax.text(0.3, 0.4, "S")

            lps_ax.set_xticks([])
            lps_ax.set_yticks([])

            plt.show()

        class volume:
            """Organizes all volume data and information. 
            Args:
                spatialRef (imref3d, optional): Imaging data orientation (axial, sagittal or coronal).
                scanRot (ndarray, optional): Array of the rotation applied to the XYZ points of the ROI.
                data (ndarray, optional): n-dimensional of the imaging data.
            Attributes:
                spatialRef (imref3d): Imaging data orientation (axial, sagittal or coronal).
                scanRot (ndarray): Array of the rotation applied to the XYZ points of the ROI.
                data (ndarray): n-dimensional of the imaging data.
            """
            def __init__(self, spatialRef=None, scanRot=None, data=None):
                self.spatialRef = spatialRef
                self.scanRot = scanRot
                self.data = data
            
            def init_transScanToModel(self, transScanToModel_value):
                self.transScanToModel = transScanToModel_value

            def update_spatialRef(self, spatialRef_value):
                self.spatialRef = spatialRef_value
            
            def update_scanRot(self, scanRot_value):
                self.scanRot = scanRot_value
            
            def update_transScanToModel(self, transScanToModel_value):
                self.transScanToModel = transScanToModel_value
            
            def update_data(self, data_value):
                self.data = data_value
            
            def convert_to_LPS(self):
                """Convert Imaging data to LPS (Left-Posterior-Superior) coordinates system.
                <https://www.slicer.org/wiki/Coordinate_systems>.
                Args:
                    ket (int): ROI index (A volume can have multiple ROIs).
                Returns:
                    None.
                """
                # flip x
                self.data = np.flip(self.data, 0)
                # flip y
                self.data = np.flip(self.data, 1)
                # to LPS
                self.data = self.data.swapaxes(0, 1) #TODO
            
            def spatialRef_from_NIFTI(self, NiftiImagePath):
                """Computes the imref3d spatialRef using a NIFTI file and
                updates the spatialRef attribute.
                Args:
                    NiftiImagePath (str): String of the NIFTI file path.
                Returns:
                    None.
                
                """
                # Loading the nifti file :
                nifti = nib.load(NiftiImagePath)
                nifti_data = self.data

                # spatialRef Creation
                pixelX = nifti.affine[0, 0]
                pixelY = nifti.affine[1, 1]
                sliceS = nifti.affine[2, 2]
                min_grid = nifti.affine[:3, 3]
                min_Xgrid = min_grid[0]
                min_Ygrid = min_grid[1]
                min_Zgrid = min_grid[2]
                size_image = np.shape(nifti_data)
                spatialRef = imref3d(size_image, abs(pixelX), abs(pixelY), abs(sliceS))
                spatialRef.XWorldLimits = (np.array(spatialRef.XWorldLimits) -
                                        (spatialRef.XWorldLimits[0] -
                                            (min_Xgrid-pixelX/2))
                                        ).tolist()
                spatialRef.YWorldLimits = (np.array(spatialRef.YWorldLimits) -
                                        (spatialRef.YWorldLimits[0] -
                                            (min_Ygrid-pixelY/2))
                                        ).tolist()
                spatialRef.ZWorldLimits = (np.array(spatialRef.ZWorldLimits) -
                                        (spatialRef.ZWorldLimits[0] -
                                            (min_Zgrid-sliceS/2))
                                        ).tolist()

                # Converting the results into lists
                spatialRef.ImageSize = spatialRef.ImageSize.tolist()
                spatialRef.XIntrinsicLimits = spatialRef.XIntrinsicLimits.tolist()
                spatialRef.YIntrinsicLimits = spatialRef.YIntrinsicLimits.tolist()
                spatialRef.ZIntrinsicLimits = spatialRef.ZIntrinsicLimits.tolist()

                # update spatialRef
                self.update_spatialRef(spatialRef)

            def convert_spatialRef(self):
                """converts the MEDimage spatialRef from RAS to LPS coordinates system.
                <https://www.slicer.org/wiki/Coordinate_systems>.
                Args:
                    None.
                Returns:
                    None.
                """
                # swap x and y data
                temp = self.spatialRef.ImageExtentInWorldX
                self.spatialRef.ImageExtentInWorldX = self.spatialRef.ImageExtentInWorldY
                self.spatialRef.ImageExtentInWorldY = temp

                temp = self.spatialRef.PixelExtentInWorldX
                self.spatialRef.PixelExtentInWorldX = self.spatialRef.PixelExtentInWorldY
                self.spatialRef.PixelExtentInWorldY = temp

                temp = self.spatialRef.XIntrinsicLimits
                self.spatialRef.XIntrinsicLimits = self.spatialRef.YIntrinsicLimits
                self.spatialRef.YIntrinsicLimits = temp

                temp = self.spatialRef.XWorldLimits
                self.spatialRef.XWorldLimits = self.spatialRef.YWorldLimits
                self.spatialRef.YWorldLimits = temp
                del temp

        class ROI:
            """Organizes all ROI data and information. 
            Args:
                indexes (Dict, optional): Dict of the ROI indexes for each ROI name.
                roi_names (Dict, optional): Dict of the ROI names.
            Attributes:
                indexes (Dict): Dict of the ROI indexes for each ROI name.
                roi_names (Dict): Dict of the ROI names.
                nameSet (Dict): Dict of the User-defined name for Structure Set for each ROI name.
                nameSetInfo (Dict): Dict of the names of the structure sets that define the areas of 
                    significance. Either 'StructureSetName', 'StructureSetDescription', 'SeriesDescription' 
                    or 'SeriesInstanceUID'.
            """
            def __init__(self, indexes=None, roi_names=None) -> None:
                self.indexes = indexes if indexes else {}
                self.roi_names = roi_names if roi_names else {}
                self.nameSet = roi_names if roi_names else {}
                self.nameSetInfo = roi_names if roi_names else {}

            def get_indexes(self, key):
                if not self.indexes or key is None:
                    return {}
                else:
                    return self.indexes[str(key)]

            def get_ROIname(self, key):
                if not self.roi_names or key is None:
                    return {}
                else:
                    return self.roi_names[str(key)]

            def get_nameSet(self, key):
                if not self.nameSet or key is None:
                    return {}
                else:
                    return self.nameSet[str(key)]

            def get_nameSetInfo(self, key):
                if not self.nameSetInfo or key is None:
                    return {}
                else:
                    return self.nameSetInfo[str(key)]

            def update_indexes(self, key, indexes):
                try: 
                    self.indexes[str(key)] = indexes
                except:
                    Warning.warn("Wrong key given in update_indexes()")

            def update_ROIname(self, key, ROIname):
                try:
                    self.roi_names[str(key)] = ROIname
                except:
                    Warning.warn("Wrong key given in update_ROIname()")

            def update_nameSet(self, key, nameSet):
                try:
                    self.nameSet[str(key)] = nameSet
                except:
                    Warning.warn("Wrong key given in update_nameSet()")

            def update_nameSetInfo(self, key, nameSetInfo):
                try:
                    self.nameSetInfo[str(key)] = nameSetInfo
                except:
                    Warning.warn("Wrong key given in update_nameSetInfo()")
            
            def convert_to_LPS(self, data):
                """
                -------------------------------------------------------------------------
                DESCRIPTION:
                This function converts the given volume to LPS coordinates system. For 
                more details please refer here : https://www.slicer.org/wiki/Coordinate_systems 
                -------------------------------------------------------------------------
                INPUTS:
                - data : given volume data in RAS to be converted to LPS
                -------------------------------------------------------------------------
                OUTPUTS:
                - data in LPS.
                -------------------------------------------------------------------------
                """
                # flip x
                data = np.flip(data, 0)
                # flip y
                data = np.flip(data, 1)
                # to LPS
                data = data.swapaxes(0, 1)

                return data

            def get_ROI_from_path(self, ROI_path, ID):
                """
                -------------------------------------------------------------------------
                DESCRIPTION:
                This function extracts all ROI data from the given path for the given
                patient ID and updates all class attributes with the new extracted data.
                This method is called only once for NIFTI formats per patient.
                -------------------------------------------------------------------------
                INPUTS:
                - ROI_path : Path where the ROI data is stored
                - ID : The ID contains patient ID and the modality type, which makes it
                possible for the method to extract the right data.
                -------------------------------------------------------------------------
                OUTPUTS:
                - NO OUTPUTS.
                -------------------------------------------------------------------------
                """
                self.indexes = {}
                self.roi_names = {}
                self.nameSet = {}
                self.nameSetInfo = {}
                roi_index = 0
                ListOfPatients = os.listdir(ROI_path)

                for file in ListOfPatients:
                    # Load the patient's ROI nifti files :
                    if file.startswith(ID) and file.endswith('nii.gz') and 'ROI' in file.split("."):
                        roi = nib.load(ROI_path + "/" + file)
                        roi_data = self.convert_to_LPS(data=roi.get_fdata())
                        roi_name = file[file.find("(")+1:file.find(")")]
                        nameSet = file[file.find("_")+2:file.find("(")]
                        self.update_indexes(key=roi_index, indexes=np.nonzero(roi_data.flatten()))
                        self.update_nameSet(key=roi_index, nameSet=nameSet)
                        self.update_ROIname(key=roi_index, ROIname=roi_name)
                        roi_index += 1