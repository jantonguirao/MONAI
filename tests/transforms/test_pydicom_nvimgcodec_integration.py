# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch

from monai.data.image_reader import PydicomReader
from monai.transforms import LoadImage, LoadImaged
from monai.utils import optional_import
from tests.test_utils import SkipIfNoModule, assert_allclose

import pydicom
import nvidia.nvimgcodec as nvimgcodec

class TestPydicomNvimgcodecIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Prepare a temporary directory for HTJ2K DICOM test data
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.htj2k_data_dir = os.path.join(cls.temp_dir.name, "CT_DICOM_HTJ2K")
        print(f"HTJ2K test data directory: {cls.htj2k_data_dir}")
        os.makedirs(cls.htj2k_data_dir, exist_ok=True)

        # Find the source CT_DICOM directory
        source_dir = Path("tests/testing_data/CT_DICOM")
        if not source_dir.exists():
            raise RuntimeError(f"Source directory {source_dir} not found for transcoding HTJ2K test data.")

        # HTJ2K transfer syntax UID to use
        htj2k_uid = "1.2.840.10008.1.2.4.201"

        cls.encoder = nvimgcodec.Encoder()
        # Encode to HTJ2K
        jpeg2k_encode_params = nvimgcodec.Jpeg2kEncodeParams()
        jpeg2k_encode_params.num_resolutions = 6
        jpeg2k_encode_params.code_block_size = (64, 64)
        jpeg2k_encode_params.bitstream_type = nvimgcodec.Jpeg2kBitstreamType.JP2
        jpeg2k_encode_params.prog_order = nvimgcodec.Jpeg2kProgOrder.LRCP
        jpeg2k_encode_params.ht = True  # Enable HTJ2K

        encode_params = nvimgcodec.EncodeParams(
            quality_type=nvimgcodec.QualityType.LOSSLESS,
            jpeg2k_encode_params=jpeg2k_encode_params,
        )

        cls.reference_data = {}

        # Transcode each CT_DICOM file to each HTJ2K transfer syntax
        for src_file in source_dir.glob("*"):
            if not src_file.is_file():
                continue
            print(f"Transcoding {src_file} to HTJ2K...")
            ds = pydicom.dcmread(str(src_file))
            
            # Decode original pixel data to numpy array
            arr = ds.pixel_array
            
            # Prepare encoding parameters
            encode_params = {
                "transfer_syntax_uid": htj2k_uid,
                "rows": ds.Rows,
                "columns": ds.Columns,
                "samples_per_pixel": getattr(ds, "SamplesPerPixel", 1),
                "bits_allocated": ds.BitsAllocated,
                "bits_stored": ds.BitsStored,
                "pixel_representation": ds.PixelRepresentation,
                "photometric_interpretation": ds.PhotometricInterpretation,
            }
            
            # Encode using nvimgcodec
            encoded = cls.encoder.encode(
                [arr], 
                codec="jpeg2k", 
                params=nvimgcodec.EncodeParams(
                    quality_type=nvimgcodec.QualityType.LOSSLESS,
                    jpeg2k_encode_params=jpeg2k_encode_params,
                )
            )
            # Create a copy of the dataset and update transfer syntax and PixelData
            ds_htj2k = ds.copy()
            ds_htj2k.file_meta.TransferSyntaxUID = htj2k_uid
            # Create encapsulated pixel data
            new_encoded_frames = [bytes(code_stream) for code_stream in encoded]
            encapsulated_pixel_data = pydicom.encaps.encapsulate(new_encoded_frames)
            ds_htj2k.PixelData = encapsulated_pixel_data

            # Save to output directory
            out_name = f"{src_file.stem}_htj2k_{htj2k_uid.split('.')[-1]}.dcm"
            out_path = os.path.join(cls.htj2k_data_dir, out_name)
            print(f"Saving HTJ2K file to {out_path}")
            ds_htj2k.save_as(out_path)

            cls.reference_data[out_path] = arr

    @classmethod
    def tearDownClass(cls):
        # Clean up the temporary directory
        cls.temp_dir.cleanup()

    def test_pydicom_htj2k_support(self):
        """Test HTJ2K support in pydicom dcmread."""
        ds = pydicom.dcmread(os.path.join(self.htj2k_data_dir, "7166_htj2k_201.dcm"))
        self.assertIsNotNone(ds)
        self.assertEqual(ds.file_meta.TransferSyntaxUID, "1.2.840.10008.1.2.4.201")
        # Check that the dataset has the correct pixel data
        self.assertIsNotNone(ds.PixelData)
        np.testing.assert_equal(ds.pixel_array, self.reference_data[os.path.join(self.htj2k_data_dir, "7166_htj2k_201.dcm")])

    def test_pydicom_nvimgcodec_htj2k_support(self):
        """Test HTJ2K support in PydicomReader with nvimgcodec."""
        reader_nvimgcodec = PydicomReader(use_nvimgcodec=True)
        reader_ref = PydicomReader(use_nvimgcodec=False)
        self.assertIsNotNone(reader_nvimgcodec)
        self.assertIsNotNone(reader_ref)
        
        # Check that the reader can decode the HTJ2K files created in setUpClass
        assert os.path.exists(self.htj2k_data_dir), f"HTJ2K test data directory {self.htj2k_data_dir} not found"
        htj2k_files = [f for f in os.listdir(self.htj2k_data_dir) if f.endswith(".dcm")]
        assert htj2k_files, "No HTJ2K DICOM files found for testing"
        for fname in htj2k_files:
            fpath = os.path.join(self.htj2k_data_dir, fname)
            try:
                img0 = reader_nvimgcodec.read(fpath)
                img1 = reader_ref.read(fpath)
            except Exception as e:
                self.fail(f"Failed to decode HTJ2K file {fname}: {e}")
            
            data0, metadata0 = reader_nvimgcodec.get_data(img0)
            data1, metadata1 = reader_ref.get_data(img1)
            np.testing.assert_equal(data0, data1)

if __name__ == "__main__":
    unittest.main(verbosity=2)