# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional

import mmengine.fileio as fileio
import numpy as np

import mmcv
from .base import BaseTransform
from .builder import TRANSFORMS


@TRANSFORMS.register_module()
class LoadImageFromFile(BaseTransform):
    """Load an image from file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv.imfrombytes`.
            See :func:`mmcv.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename)
                img_bytes = file_client.get(filename)
            else:
                img_bytes = fileio.get(
                    filename, backend_args=self.backend_args)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            # print("succccccssssss111",img.shape)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename}'
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        # print("result======ressult", results)
        # print("dddddddddd")
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', ")

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str


@TRANSFORMS.register_module()
class LoadAnnotations(BaseTransform):
    """Load and process the ``instances`` and ``seg_map`` annotation provided
    by dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            'instances':
            [
                {
                # List of 4 numbers representing the bounding box of the
                # instance, in (x1, y1, x2, y2) order.
                'bbox': [x1, y1, x2, y2],

                # Label of image classification.
                'bbox_label': 1,

                # Used in key point detection.
                # Can only load the format of [x1, y1, v1,…, xn, yn, vn]. v[i]
                # means the visibility of this keypoint. n must be equal to the
                # number of keypoint categories.
                'keypoints': [x1, y1, v1, ..., xn, yn, vn]
                }
            ]
            # Filename of semantic or panoptic segmentation ground truth file.
            'seg_map_path': 'a/b/c'
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
            # In (x1, y1, x2, y2) order, float type. N is the number of bboxes
            # in np.float32
            'gt_bboxes': np.ndarray(N, 4)
             # In np.int64 type.
            'gt_bboxes_labels': np.ndarray(N, )
             # In uint8 type.
            'gt_seg_map': np.ndarray (H, W)
             # with (x, y, v) order, in np.float32 type.
            'gt_keypoints': np.ndarray(N, NK, 3)
        }

    Required Keys:

    - instances

      - bbox (optional)
      - bbox_label
      - keypoints (optional)

    - seg_map_path (optional)

    Added Keys:

    - gt_bboxes (np.float32)
    - gt_bboxes_labels (np.int64)
    - gt_seg_map (np.uint8)
    - gt_keypoints (np.float32)

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
            Defaults to True.
        with_label (bool): Whether to parse and load the label annotation.
            Defaults to True.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Defaults to False.
        with_keypoints (bool): Whether to parse and load the keypoints
            annotation. Defaults to False.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv.imfrombytes`.
            See :func:`mmcv.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(
        self,
        with_bbox: bool = True,
        with_label: bool = True,
        with_seg: bool = False,
        with_keypoints: bool = False,
        imdecode_backend: str = 'cv2',
        file_client_args: Optional[dict] = None,
        *,
        backend_args: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_seg = with_seg
        self.with_keypoints = with_keypoints
        self.imdecode_backend = imdecode_backend

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def _load_bboxes(self, results: dict) -> None:
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        gt_bboxes = []
        for instance in results['instances']:
            gt_bboxes.append(instance['bbox'])
        results['gt_bboxes'] = np.array(
            gt_bboxes, dtype=np.float32).reshape(-1, 4)

    def _load_labels(self, results: dict) -> None:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        gt_bboxes_labels = []
        for instance in results['instances']:
            gt_bboxes_labels.append(instance['bbox_label'])
        results['gt_bboxes_labels'] = np.array(
            gt_bboxes_labels, dtype=np.int64)

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """
        if self.file_client_args is not None:
            file_client = fileio.FileClient.infer_client(
                self.file_client_args, results['seg_map_path'])
            img_bytes = file_client.get(results['seg_map_path'])
        else:
            img_bytes = fileio.get(
                results['seg_map_path'], backend_args=self.backend_args)

        results['gt_seg_map'] = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze()

    def _load_kps(self, results: dict) -> None:
        """Private function to load keypoints annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded keypoints annotations.
        """
        gt_keypoints = []
        for instance in results['instances']:
            gt_keypoints.append(instance['keypoints'])
        results['gt_keypoints'] = np.array(gt_keypoints, np.float32).reshape(
            (len(gt_keypoints), -1, 3))

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation and keypoints annotations.
        """

        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        if self.with_seg:
            self._load_seg_map(results)
        if self.with_keypoints:
            self._load_kps(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'with_keypoints={self.with_keypoints}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str

# from some_segmentation_library import SegmentationModel  # Replace with your actual model import   
from torchvision.models.segmentation import deeplabv3_resnet101   
import torch
# import numpy as np
# import mmcv
# from typing import Optional

# from mmengine.dataset import BaseTransform
@TRANSFORMS.register_module()
class LoadSegLogitsFromFile(BaseTransform):
    """Load segmentation logits from an image file and apply softmax.

    Required Keys:

    - img_path

    Modified Keys:

    - seg_logits
    - img_shape
    - ori_shape

    Args:
        model (SegmentationModel): A pre-trained segmentation model.
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is a uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type.
            Defaults to 'cv2'.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 ignore_empty: bool = False) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        
        # Initialize the DeepLabV3 model
        self.model = deeplabv3_resnet101(pretrained=True)
        self.model.eval()  # Set to evaluation mode
       

    def transform(self, results: dict) -> Optional[dict]:
        """Load image and get segmentation logits.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains segmentation logits, softmax entropy,
            softmax distance, and meta information.
        """

        filename = results['img_path']
        try:
            img = mmcv.imread(filename, flag=self.color_type, backend=self.imdecode_backend)
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            img = img.unsqueeze(0).repeat(1, 1, 1, 1)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e

        assert img is not None, f'failed to load image: {filename}'

        # Process through the segmentation model
        seg_logits = self.model(img)['out']  # Extracting data from the result

        # Apply softmax to obtain probabilities
        seg_probs = torch.softmax(seg_logits, dim=1)

        # Calculate softmax entropy
        seg_entropy = self.softmax_entropy(seg_logits)

        # Calculate softmax distance
        seg_distance = self.softmax_distance(seg_probs)

        if self.to_float32:
            seg_probs = seg_probs.astype(np.float32)
            seg_entropy = seg_entropy.astype(np.float32)
            seg_distance = seg_distance.astype(np.float32)

        results['seg_logits'] = seg_probs
        results['seg_entropy'] = seg_entropy
        results['seg_distance'] = seg_distance
        results['img_shape'] = seg_probs.shape[2:]  # Assuming seg_probs is in NxCxHxW
        results['ori_shape'] = seg_probs.shape[2:]
        return results

    @staticmethod
    def softmax_entropy(logits):
        exp_logits = torch.exp(logits - torch.max(logits, dim=1, keepdim=True)[0])  # Stability adjustment
        softmax_probs = exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)  # Normalize to get probabilities

        epsilon = 1e-10  # Small value to avoid log(0)
        log_probs = torch.log(softmax_probs + epsilon)  # log(P)
        entropy = -torch.sum(softmax_probs * log_probs, dim=1)  # Sum across classes (dim=1)

        return entropy  # Shape: (batch_size, height, width)

    @staticmethod
    def softmax_distance(probs):
        # Calculate the distance between the highest and second highest probabilities
        top2_probs, _ = torch.topk(probs, 2, dim=1)
        distance = top2_probs[:, 0, :, :] - top2_probs[:, 1, :, :]
        return distance

    def __repr__(self):
        return (f'{self.__class__.__name__}(ignore_empty={self.ignore_empty}, '
                f'to_float32={self.to_float32}, '
                f"color_type='{self.color_type}', "
                f"imdecode_backend='{self.imdecode_backend}', "
                f'model={self.model})')