from numpy.typing import NDArray
from typing import List, Any
from enum import Enum
from dataclasses import dataclass
import numpy as np
import cv2
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy import linalg

from smooth_gradient_outpaint.config import OutpainterConfig


class EdgeDetector:
    def __init__(self):
        self.config = OutpainterConfig().config
        self.THRESHOLD1 = self.config["edge_detector"]["parameters"]["low_threshold"]
        self.THRESHOLD2 = self.config["edge_detector"]["parameters"]["high_threshold"]
        self.BLUR_KERNEL_SIZE_TOP = self.config["edge_detector"]["blur"]["parameters"]["top"]["size"]
        self.BLUR_KERNEL_SIZE_BOTTOM = self.config["edge_detector"]["blur"]["parameters"]["bottom"]["size"]
        self.BLUR_KERNEL_IMAGE_SIZE_TOP = self.config["edge_detector"]["blur"]["parameters"]["top"]["image_size"]
        self.BLUR_KERNEL_IMAGE_SIZE_BOTTOM = self.config["edge_detector"]["blur"]["parameters"]["bottom"]["image_size"]
        self.BLUR_SIGMA = self.config["edge_detector"]["parameters"]["sigma"]

    def get_blur_kernel_size(self, img: NDArray[List[np.uint8]]) -> tuple[int, ...]:
        """
        Get the kernel size for Gaussian blur based on the image size using the two extreme values - top and bottom.
        Interpolate everything in between.
        :param img: image to be processed
        :return: kernel size for Gaussian blur
        """
        img_height, img_width = img.shape[:2]
        larger_img_size = max(img_height, img_width)
        top_image_size = self.BLUR_KERNEL_IMAGE_SIZE_TOP
        bottom_image_size = self.BLUR_KERNEL_IMAGE_SIZE_BOTTOM
        top_blur_size = self.BLUR_KERNEL_SIZE_TOP
        bottom_blur_size = self.BLUR_KERNEL_SIZE_BOTTOM
        if larger_img_size >= top_image_size:
            blur = tuple([top_blur_size] * 2)
        elif top_image_size > larger_img_size > bottom_image_size:
            blur = (tuple(
                [int(np.floor(
                    bottom_blur_size + (top_blur_size - bottom_blur_size) / (top_image_size - bottom_image_size) *
                    (larger_img_size - bottom_image_size)))] * 2))
        else:
            blur = tuple([bottom_blur_size] * 2)

        return blur

    def detect_edges(self, img: NDArray[List[np.uint8]]) -> NDArray[np.uint8]:
        """
        Detect edges in image using Canny edge detector
        :param img: image to detect edges in (ndarray[list[np.uint8]])
        :return: mask of edges
        """
        blur_kernel_size = self.get_blur_kernel_size(img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, blur_kernel_size, self.BLUR_SIGMA)
        edges = cv2.Canny(img_blur, self.THRESHOLD1, self.THRESHOLD2)
        return edges


class Location(Enum):
    LEFT = 0
    RIGHT = 1
    TOP = 2
    BOTTOM = 3


def find_largest_area_with_smooth_background(edges: NDArray[np.uint8], direction: Location) -> int:
    """
    Find the largest padding (zone without edges) on one side of the image
    :param mask: mask of edges
    :param direction: location of the background area which width/height should be computed (Enum BackgroundLocation)
    :return: size in pixels of the largest background area in the specified direction
    """
    if (direction == Location.LEFT or
            direction == Location.RIGHT):
        mask = np.any(edges == 255, axis=0)
    else:
        mask = np.any(edges == 255, axis=1)

    smooth_background_size = 0

    if (direction == Location.LEFT or
            direction == Location.TOP):
        for val in mask:
            if not val:
                smooth_background_size += 1
            else:
                break
    else:
        for val in mask[::-1]:
            if not val:
                smooth_background_size += 1
            else:
                break

    return smooth_background_size


def find_largest_smooth_background_dimensions(
        image: NDArray[List[np.uint8]], sides: List[Location] = None) -> dict[Location, int]:
    """
    Find the largest padding (zone without edges) on each side of the image
    :param image: RGB image to be processed
    :param sides: list of sides to check
    :return: dict with largest padding for each side
    """
    if sides is None:
        sides = [Location.LEFT, Location.RIGHT, Location.TOP, Location.BOTTOM]
    edges_detector = EdgeDetector()
    edges_mask = edges_detector.detect_edges(image)
    return {side: find_largest_area_with_smooth_background(edges_mask, side) for side in sides}


def compute_outpaint_area_dimensions(
    image: NDArray[List[np.uint8]],
    aspect_ratio: float,
    object_of_interest_bbox: dict[str, int] = None,
    adjust_for_ooi_offset: bool = True,
) -> dict[Location, int]:
    """
    Compute the dimensions of the outpainted area(s) given the specified aspect ratio
    :param image: RGB image to be processed
    :param aspect_ratio: desired aspect ratio
    :param object_of_interest_bbox: bounding box of the object of interest
      Example: {'x': 0, 'y': 0, 'width': 100, 'height': 100}
    :param adjust_for_ooi_offset: whether to adjust the outpainting to the object of interest offset
    :return: dict with dimensions of the outpainted regions
    """
    image_height, image_width = image.shape[:2]
    image_aspect_ratio = image_width / image_height
    if image_aspect_ratio < aspect_ratio:
        new_width = int(image_height * aspect_ratio)
        new_height = image_height
    else:
        new_width = image_width
        new_height = int(image_width / aspect_ratio)

    backgrounds = find_largest_smooth_background_dimensions(image)
    outpaintings = dict.fromkeys(backgrounds.keys(), 0)
    ooi_box_center_x = (
        object_of_interest_bbox["x"] + object_of_interest_bbox["width"] // 2
    )
    ooi_box_center_y = (
        object_of_interest_bbox["y"] + object_of_interest_bbox["height"] // 2
    )
    if new_width > image_width:

        offset_horizontal = image_width // 2 - ooi_box_center_x
        if (
            Location.LEFT in backgrounds
            and Location.RIGHT in backgrounds
            and backgrounds[Location.LEFT] > 0
            and backgrounds[Location.RIGHT] > 0
        ):
            outpaintings[Location.LEFT] = (new_width - image_width) // 2
            outpaintings[Location.RIGHT] = new_width - image_width - outpaintings[Location.LEFT]
            if adjust_for_ooi_offset:
                outpaintings[Location.LEFT] += offset_horizontal
                outpaintings[Location.RIGHT] -= offset_horizontal
        elif Location.LEFT in backgrounds and backgrounds[Location.LEFT] > 0:
            outpaintings[Location.LEFT] = new_width - image_width
        elif Location.RIGHT in backgrounds and backgrounds[Location.RIGHT] > 0:
            outpaintings[Location.RIGHT] = new_width - image_width
        else:
            outpaintings[Location.LEFT] = 0
            outpaintings[Location.RIGHT] = 0

        outpaintings[Location.TOP] = 0
        outpaintings[Location.BOTTOM] = 0
    else:
        offset_vertical = image_height // 2 - ooi_box_center_y
        if (
            Location.TOP in backgrounds
            and Location.BOTTOM in backgrounds
            and backgrounds[Location.TOP] > 0
            and backgrounds[Location.BOTTOM] > 0
        ):
            outpaintings[Location.TOP] = (new_height - image_height) // 2
            outpaintings[Location.BOTTOM] = new_height - image_height - outpaintings[Location.TOP]
            if adjust_for_ooi_offset:
                outpaintings[Location.TOP] += offset_vertical
                outpaintings[Location.BOTTOM] -= offset_vertical
        elif Location.TOP in backgrounds and backgrounds[Location.TOP] > 0:
            outpaintings[Location.TOP] = new_height - image_height
        elif Location.BOTTOM in backgrounds and backgrounds[Location.BOTTOM] > 0:
            outpaintings[Location.BOTTOM] = new_height - image_height
        else:
            outpaintings[Location.TOP] = 0
            outpaintings[Location.BOTTOM] = 0

        outpaintings[Location.LEFT] = 0
        outpaintings[Location.RIGHT] = 0

    return outpaintings


@dataclass
class VerticalGradientRegion:
    col_idx: int
    row_idx_start: int
    row_idx_end: int
    row_idx_visible_start: int
    row_idx_visible_end: int
    row_idx_darkest: int
    darkest_intensity_delta: float
    intensity_gradient: float
    region_coverage: float
    is_shadow: bool


@dataclass
class ShadowRegionsFeatures:
    idx_col_arr: NDArray[int]
    idx_top_arr: NDArray[int]
    idx_top_viz_arr: NDArray[int]
    idx_darkest_arr: NDArray[int]
    idx_bottom_viz_arr: NDArray[int]
    idx_bottom_arr: NDArray[int]
    intensity_darkest_arr: NDArray[float]
    intensity_viz_thresh_arr: NDArray[float]


class ShadowDetector:
    def __init__(self, add_only_shadow_regions: bool = False,
                 enable_intra_column_discrepancy_check: bool = True,
                 enable_inter_column_discrepancy_check: bool = True):
        """
        Initialize the ShadowDetector class
        :param add_only_shadow_regions: whether to add only shadow regions or to add also transition regions as well
        :param enable_intra_column_discrepancy_check: whether to enable intra column discrepancy check.
          Note: The intra column discrepancy check is performed on the shadow region of the current column which is
           compared with the transient region of the current column only.
           The inter column discrepancy check compares the shadow region of the current column with the shadow region of
            the most recently processed column.
        :param enable_inter_column_discrepancy_check: whether to enable inter column discrepancy check
        """
        self.add_only_shadow_regions = add_only_shadow_regions
        self.enable_intra_column_discrepancy_check = enable_intra_column_discrepancy_check
        self.enable_inter_column_discrepancy_check = enable_inter_column_discrepancy_check
        self.config = OutpainterConfig().config
        self.PERCENT_UNIFORMITY_THRESHOLD = self.config["shadow_detector"]["parameters"]["percent_uniformity_threshold"]
        self.BLUR_KERNEL_SIZE = tuple([self.config["shadow_detector"]["blur"]["parameters"]["size"]] * 2)
        self.BLUR_SIGMA = self.config["shadow_detector"]["blur"]["parameters"]["sigma"]
        self.GRADIENT_KERNEL_SIZE = self.config["shadow_detector"]["parameters"]["gradient_kernel_size"]
        self.SLIDING_WINDOW_SIZE = self.config["shadow_detector"]["parameters"]["sliding_window_size"]
        self.NUMBER_OF_QUANTA = self.config["shadow_detector"]["parameters"]["number_of_quanta"]
        self.REGION_EXPLORATION_FACTOR_BELOW = (
            self.config)["shadow_detector"]["parameters"]["region_exploration_factor_below"]
        self.REGION_EXPLORATION_FACTOR_ABOVE = (
            self.config)["shadow_detector"]["parameters"]["region_exploration_factor_above"]
        self.REGION_INTERPOLATION_FACTOR_BELOW = (
            self.config)["shadow_detector"]["parameters"]["region_interpolation_factor_below"]
        self.REGION_INTERPOLATION_FACTOR_ABOVE = (
            self.config)["shadow_detector"]["parameters"]["region_interpolation_factor_above"]
        self.VISIBLE_SHADOW_QUANTILE = self.config["shadow_detector"]["parameters"]["visible_shadow_quantile"]
        self.MAX_NUMBER_OF_REGIONS = self.config["shadow_detector"]["parameters"]["max_number_of_regions"]
        self.DISCREPANCY_THRESHOLD_FACTOR = self.config["shadow_detector"]["parameters"]["discrepancy_threshold_factor"]
        self.GRADIENT_KERNEL_SCALING_FACTOR = self.compute_scaling_factor_for_gradient_kernel()

    def compute_scaling_factor_for_gradient_kernel(self):
        """
        In order to be a "proper" derivative estimator, the 3x3 Sobel should be scaled by a factor of 1/8:
        sob3x3 = 1/8 * [ 1 2 1 ]' * [1 0 -1]
        and each larger kernel needs to be scaled by an additional factor of 1/16.
        This is because the smoothing kernels are not normalised:
        sob5x5 = 1/16 * conv2( [ 1 2 1 ]' * [1 2 1], sob3x3 )
        sob7x7 = 1/16 * conv2( [ 1 2 1 ]' * [1 2 1], sob5x5 )
        ...
        sobnxn = 1/16 * conv2( [ 1 2 1 ]' * [1 2 1], sob(n-2)x(n-2) )
        """
        factor = 8 * pow(16, self.GRADIENT_KERNEL_SIZE // 2 - 1)
        return factor

    def find_vertical_gradient_regions(self, gradient_background: NDArray[np.uint8],
                                       background_columns: List[int]) -> List[VerticalGradientRegion]:
        """
        Function to find where the gradient background changes from darker to lighter in vertical direction as well as
        identify shadow regions
        The Algorithm:
        1) Convert the RGB image to grayscale
        2) Apply Gaussian blur to the grayscale image
        3) Compute the gradient along the Y axis using Sobel Y transform
        4) Slide a window with size S vertically over the gradient background and compute the covering intensity
        threshold I_0.
        5) Find the positions of the window with size S yielding maximum covering intensity gradient
        6) Enlarge the window to size S_new such that the upsized window will provide a continuous cover for all windows
         with size S obeying  | g_i - g_j | >= PERCENT_THRESHOLD for each pair  i and j within S_new
        7) Decide if the window obtained in step 6) captures gradient transition region or a shadow:
          compute a score involving the area of a region equal to the width of the window situated above the window.
          If the score is positive the window captures the gradient transition region, otherwise it captures a shadow.
        8) mark the position and width of the last upsized window as "occupied".
        9) repeat 4)-8) looking for another window in unoccupied area.
        Assumptions to be validated:
        i) in case of a single transition region from dark to light a shadow could exist only below
        the transition region.
        ii) there can be more than one shadow region below the transition region from dark to light.

        Args:
        - gradient_background (NDArray[List[np.uint8]]) - numpy array containing a single channel or grayscale level
          of an image of an area with gradient background
        - background_columns int - column index from the gradient background region to be processed

        Returns:
        - a list of VerticalGradientRegion instances
        """

        blurred_background = cv2.GaussianBlur(gradient_background, self.BLUR_KERNEL_SIZE, self.BLUR_SIGMA)
        vert_grad = self.get_vertical_gradient(blurred_background)
        regions = list()
        for col in background_columns:
            vert_grad_col = vert_grad[:, col]
            vert_intensity_col = blurred_background[:, col]
            self.process_current_column(vert_grad_col, vert_intensity_col, regions, col)
        return regions

    def process_current_column(self, vert_grad_col: NDArray[np.floating], vert_intensity_col: NDArray[np.uint8],
                               regions: List[VerticalGradientRegion], col: int):
        """
        Process the current column of the gradient background
        :param vert_grad_col: the vertical gradient component for a specific column
        :param vert_intensity_col: the intensity component for a specific column
        :param regions: list of shadow and transition regions
        :param col: the column index
        """
        regions_for_cur_col = list()
        windows_to_be_excluded = []

        shadow_region_added, transient_region_added = (
            self.try_to_add_shadow_and_transient_regions_for_cur_column(
                col, vert_grad_col, vert_intensity_col, regions_for_cur_col, windows_to_be_excluded))

        if transient_region_added is not True:
            # add a transient region to the list of regions for current column only if it was not found so far
            self.add_transient_region_for_cur_column(col, shadow_region_added, regions_for_cur_col,
                                                     vert_grad_col, windows_to_be_excluded)

        self.sanitize_and_add_the_current_regions_to_processed_regions(regions_for_cur_col, regions, shadow_region_added)

    def try_to_add_shadow_and_transient_regions_for_cur_column(
            self, col, vert_grad_col: NDArray[np.floating], vert_intensity_col: NDArray[np.uint8],
            regions_for_cur_col: List[VerticalGradientRegion],
            windows_to_be_excluded: List[tuple[int, int]]) -> tuple[bool, bool]:
        """
        Add a shadow region for the current column and maybe a transient region with the highest gradient value
        :param col: the column index
        :param vert_grad_col: the vertical gradient component for a specific column
        :param vert_intensity_col: the intensity component for a specific column
        :param regions_for_cur_col: list of shadow and transition regions for the current column
        :param windows_to_be_excluded: list of excluded windows
        :return: True if the shadow region was added, True if the transient region was added
        """
        is_shadow_region = False
        transient_region_count = 0
        transient_region_added = False
        while not is_shadow_region and transient_region_count < self.MAX_NUMBER_OF_REGIONS:
            grad, cov, idx_top, idx_bottom = \
                self.find_max_covering_intensity_gradient_and_coverage(
                    vert_grad_col, windows_to_be_excluded=windows_to_be_excluded)

            is_shadow_region, idx_top_new, idx_bottom_new = \
                self.compute_size_of_shadow_region(vert_grad_col, (idx_top, idx_bottom))

            idx_visible_top = idx_top_new
            idx_visible_bottom = idx_bottom_new
            idx_darkest_shadow = -1
            darkest_shadow_delta = 0.0
            if is_shadow_region:
                idx_visible_top, idx_visible_bottom, idx_darkest_shadow, darkest_shadow_delta = (
                    self.estimate_visible_shadow_size(
                        vert_intensity_col, idx_top_new, idx_bottom_new, idx_bottom - idx_top))

            self.update_windows_to_be_excluded(windows_to_be_excluded, idx_top_new, idx_bottom_new)

            if transient_region_added is not True or is_shadow_region is True:
                regions_for_cur_col.append(VerticalGradientRegion(
                    col, idx_top_new, idx_bottom_new, idx_visible_top, idx_visible_bottom,
                    idx_darkest_shadow, darkest_shadow_delta, grad, cov, is_shadow_region))
                if is_shadow_region is not True:
                    transient_region_added = True

            transient_region_count += 1

        return is_shadow_region, transient_region_added

    def add_transient_region_for_cur_column(
            self, col, shadow_region_added, regions_for_cur_col, vert_grad_col,
            windows_to_be_excluded) -> bool:
        """
        Helper method to add a transient region to the list of regions for the current column
        :param shadow_region_added: boolean indicating if the current region is a shadow region
        :param regions_for_cur_col: list of shadow and transition regions for the current column
        :param vert_grad_col: the vertical gradient component for a specific column
        :param windows_to_be_excluded: list of excluded windows
        :param col: the column index
        :return: True if the transient region was added
        """
        grad = 0.0
        cov = 0.0
        idx_top = -1
        idx_bottom = -1
        idx_top_new = -1
        idx_bottom_new = -1
        last_region_is_shadow_region = shadow_region_added
        shadow_region_count = 0
        # find the transient region with the highest gradient value
        while shadow_region_added and last_region_is_shadow_region and shadow_region_count < self.MAX_NUMBER_OF_REGIONS:

            grad, cov, idx_top, idx_bottom = \
                self.find_max_covering_intensity_gradient_and_coverage(
                    vert_grad_col, windows_to_be_excluded=windows_to_be_excluded)

            last_region_is_shadow_region, idx_top_new, idx_bottom_new = \
                self.compute_size_of_shadow_region(vert_grad_col, (idx_top, idx_bottom))

            self.update_windows_to_be_excluded(windows_to_be_excluded, idx_top, idx_bottom)

            shadow_region_count += 1

        if shadow_region_added and not last_region_is_shadow_region:
            # add the transient region with max gradient value
            regions_for_cur_col.append(VerticalGradientRegion(
                col, idx_top_new, idx_bottom_new, idx_top, idx_bottom,
                -1, -1, grad, cov, False))
            return True
        return False

    def sanitize_and_add_the_current_regions_to_processed_regions(
            self, regions_for_cur_col: List[VerticalGradientRegion], processed_regions: List[VerticalGradientRegion],
            shadow_region_added: bool):
        """
        Sanitize the current regions and add them to the processed regions
        :param regions_for_cur_col: list of shadow and transition regions for the current column
        :param processed_regions: list of shadow and transition regions for all columns
        :param shadow_region_added: boolean indicating if the current region is a shadow region
        """
        if self.enable_intra_column_discrepancy_check is True and self.enable_inter_column_discrepancy_check is True:
            if (self.invalid_shadow_region_present(regions_for_cur_col) is not True and
                    shadow_region_added and
                    self.inter_column_region_discrepancy_present(regions_for_cur_col, processed_regions) is not True):
                if self.add_only_shadow_regions is True:
                    processed_regions.extend([region for region in regions_for_cur_col if region.is_shadow])
                else:
                    processed_regions.extend(regions_for_cur_col)
            else:
                # if the shadow region is corrupted, add only the transition region with max gradient value
                if self.add_only_shadow_regions is False:
                    processed_regions.extend([region for region in regions_for_cur_col if region.is_shadow is False])

        elif self.enable_intra_column_discrepancy_check is True:
            if shadow_region_added is True and self.invalid_shadow_region_present(regions_for_cur_col) is not True:
                if self.add_only_shadow_regions is True:
                    processed_regions.extend([region for region in regions_for_cur_col if region.is_shadow])
                else:
                    processed_regions.extend(regions_for_cur_col)
            else:
                # if the shadow region is corrupted, add only the transition region with max gradient value
                if self.add_only_shadow_regions is False:
                    processed_regions.extend([region for region in regions_for_cur_col if region.is_shadow is False])

        elif self.enable_inter_column_discrepancy_check is True:
            if (shadow_region_added and
                    self.inter_column_region_discrepancy_present(regions_for_cur_col, processed_regions) is not True):
                if self.add_only_shadow_regions is True:
                    processed_regions.extend([region for region in regions_for_cur_col if region.is_shadow])
                else:
                    processed_regions.extend(regions_for_cur_col)
            else:
                # if the shadow region is corrupted, add only the transition region with max gradient value
                if self.add_only_shadow_regions is False:
                    processed_regions.extend([region for region in regions_for_cur_col if region.is_shadow is False])
        else:
            processed_regions.extend(regions_for_cur_col)

    @staticmethod
    def invalid_shadow_region_present(column_regions: List[VerticalGradientRegion]) -> bool:
        """
        Check if the shadow regions of the current column are corrupted - that would be the case if the shadow region is
         before the transition region with max gradient
        :param column_regions: list of shadow and transition regions for the current column
        :return: True if the shadow regions are corrupted
        """
        shadow_region_start = None
        shadow_region_end = None
        transition_region_start = None
        transition_region_end = None
        for region in column_regions:
            if region.is_shadow:
                shadow_region_start = region.row_idx_start
                shadow_region_end = region.row_idx_end
            else:
                transition_region_start = region.row_idx_start
                transition_region_end = region.row_idx_end
        if (shadow_region_start is not None and
                transition_region_start is not None and
                shadow_region_end is not None and
                transition_region_end is not None):
            if shadow_region_end < transition_region_start:
                return True
            else:
                return False

        return False

    def inter_column_region_discrepancy_present(
            self, current_column_regions: List[VerticalGradientRegion],
            previous_column_regions: List[VerticalGradientRegion]) -> bool:
        """
        Check if the region discrepancy is present between two consecutive columns
        :param current_column_regions: List[VerticalGradientRegion]
        :param previous_column_regions: List[VerticalGradientRegion]
        :return: bool, whether the region discrepancy is present
        """

        if not current_column_regions:
            return False

        current_region = None
        # get the shadow region of the current column
        for reg in current_column_regions:
            if reg.is_shadow:
                current_region = reg
                break

        if not current_region:
            return False

        # select the shadow region of the most recent previous column
        if len(previous_column_regions) > 0:
            i = -1
            while previous_column_regions[i].is_shadow is False and i > -len(previous_column_regions):
                i -= 1
            previous_region = previous_column_regions[i]
            if previous_region.is_shadow is False:
                return False
        else:
            return False

        if previous_region is not None:
            d1 = current_region.row_idx_visible_start - previous_region.row_idx_visible_start
            d2 = previous_region.row_idx_visible_end - previous_region.row_idx_visible_start
            d3 = current_region.row_idx_visible_end - previous_region.row_idx_visible_end
            if abs(d1 / d2) > self.DISCREPANCY_THRESHOLD_FACTOR or abs(d3 / d2) > self.DISCREPANCY_THRESHOLD_FACTOR:
                # there is too large gap between two consecutive regions
                # we will assume that the current region data is invalid
                return True
        return False

    def find_covering_non_negative_gradient_and_coverage(
            self, background_vert_gradient_col: NDArray[np.floating], window: tuple[int, int]) -> tuple[float, float]:
        """
        Find the value of the intensity gradient that covers significant portion of the pixels within the specified
         window. Here the key is how we determine what number of pixels is significant. The method is based on
         discretizing the intensity into predefined number of quanta and compute the minimum percentage of pixels
         which includes the max intensity gradient.
        That is, if we denote with G the intensity gradient , the function finds such value g of the intensity
        gradient G that dP(0 < G <= g | W = window)/dg is maximal, where P(0 < G <= g | W=window) is the percentage
         of pixels with positive intensity gradient smaller or equal to g within the specified window.

        Args:
        - background_vert_gradient_col (NDArray[np.uint8]) - numpy array containing the vertical gradient
        of a given background column
        - window (tuple[int, int]) - tuple containing the top and bottom row indices of the window

        Returns:
            the value of the intensity gradient g which maximizes the gradient of the cumulative distribution function
            P(0 < G <= g | W = window) and the cover value - that is the value of P(0 < G <= g | W = window) at g.
        """
        background_vert_gradient_win = np.copy(background_vert_gradient_col[window[0]:window[1]])
        max_gradient = background_vert_gradient_win.max()
        min_gradient = background_vert_gradient_win.min()
        if max_gradient <= 0.0:
            return 0.0, 0.0

        if max_gradient == min_gradient:
            return max_gradient, 1.0

        low_threshold = 0.0
        if min_gradient > 0.0:
            low_threshold = min_gradient
        else:
            background_vert_gradient_win[background_vert_gradient_win < 0.0] = 0.0
        gradient_delta = (max_gradient - low_threshold) / self.NUMBER_OF_QUANTA

        # algorithm for computing the covering gradient and coverage using cumulative histogram and quantization
        # of the intensity gradient values
        bins = np.arange(low_threshold, max_gradient+gradient_delta, gradient_delta)

        # quantized_data = np.digitize(background_vert_gradient_win, bins)-1
        hist, bins = np.histogram(background_vert_gradient_win, bins=bins)
        cumulative_hist = np.cumsum(hist[::-1])[::-1]
        # cumulative_hist[0] = cumulative_hist[1]
        diff = np.diff(cumulative_hist)
        max_diff_index = np.argmax(diff)
        grad = float(bins[max_diff_index])
        cov = float(cumulative_hist[max_diff_index]) / (window[1] - window[0])

        # Note: the code above is equivalent to the algorithm below
        #
        # levels = list()
        # counts = list()
        # diff = list()
        # for cur_quantum in range(0, self.NUMBER_OF_QUANTA):
        #     cur_level = low_threshold + cur_quantum * gradient_delta
        #     cur_count = np.sum(np.greater(background_vert_gradient_win, cur_level))
        #     levels.append(cur_level)
        #     counts.append(cur_count)
        #     diff.append(cur_count - counts[cur_quantum - 1] if cur_quantum > 0 else 0)
        #
        # max_diff_index = np.argmax(diff)
        # grad = levels[max_diff_index]
        # cov = counts[max_diff_index] / (window[1] - window[0])
        #
        return grad, cov

    def get_vertical_gradient(self, gradient_background_blur: NDArray[List[np.uint8]]) -> NDArray[np.floating]:
        """
        Get the vertical gradient of the background image
        :param gradient_background_blur: the gradient background grayscale blurred
        :return: the vertical gradient of the background image
        """
        sobel_y = cv2.Sobel(src=gradient_background_blur, ddepth=cv2.CV_64F, dx=0, dy=1,
                            ksize=self.GRADIENT_KERNEL_SIZE)
        return sobel_y / self.GRADIENT_KERNEL_SCALING_FACTOR

    @staticmethod
    def window_is_excluded(windows_to_be_excluded: List[tuple[int, int]], sliding_window: tuple[int, int]) -> bool:
        """
        Check if there is an overlap between the sliding window and the windows to be excluded
        """
        if windows_to_be_excluded:
            for window in windows_to_be_excluded:
                if window[1] > sliding_window[0] >= window[0] or window[1] > sliding_window[1] > window[0]:
                    return True
        return False

    @staticmethod
    def update_windows_to_be_excluded(windows_to_be_excluded, idx_top, idx_bottom):
        """
        Update the list of windows to be excluded
        :param windows_to_be_excluded: list of excluded windows
        :param idx_top: top index of the window to be excluded
        :param idx_bottom: bottom index of the window to be excluded
        :return: updated list of excluded windows
        Note: no need for binary search since the list is expected to be small (< 10)
        """
        if not windows_to_be_excluded:
            windows_to_be_excluded.append((idx_top, idx_bottom))
            return windows_to_be_excluded

        if idx_bottom < windows_to_be_excluded[0][0]:
            windows_to_be_excluded.insert(0, (idx_top, idx_bottom))
            return windows_to_be_excluded

        if idx_top >= windows_to_be_excluded[-1][1]:
            windows_to_be_excluded.append((idx_top, idx_bottom))
            return windows_to_be_excluded

        for idx, win in enumerate(windows_to_be_excluded):
            if win[0] < idx_top:
                if idx + 1 < len(windows_to_be_excluded):
                    if windows_to_be_excluded[idx + 1][0] > idx_bottom:
                        windows_to_be_excluded.insert(idx + 1, (idx_top, idx_bottom))
                        break

        return windows_to_be_excluded

    def find_max_covering_intensity_gradient_and_coverage(
            self, vertical_gradient_col: NDArray[np.floating], windows_to_be_excluded: List[tuple[int, int]] = None)\
            -> tuple[float, float, int, int]:
        """
        Helper method to run the find_covering_intensity_gradient_and_coverage method on the given input
        :param vertical_gradient_col: the vertical gradient of the grayscale background with applied Gaussian blur
            for specific column
        :param windows_to_be_excluded: a sorted from top to bottom list with windows to be excluded from
         the search inside the current column
        :return (float) the covering gradient value of the max gradient region,
                (float) the coverage value of the max gradient region,
                (int) the top index of the max gradient region,
                (int) the bottom index of the max gradient region
        """

        """
        The Algorithm:
        1) create a sliding window with fixed width self.SLIDING_WINDOW_SIZE which slides from row 0 to
         row len(height) - self.SLIDING_WINDOW_SIZE.
        2) start moving with a single pixel resolution the sliding window with width self.SLIDING_WINDOW_SIZE
        from top of the column toward the bottom of the column.
        3) with the current position of the sliding window estimate the value which covers
         significant portion of the pixel intensity gradient values using the predefined quantization range.
          store the current value of the covering intensity gradient with the current position of the sliding window
           in a list.
        4) find the maximal value of the covering intensity gradient and sliding window position which corresponds to
         it. we will denote this sliding window position as the peak window position and the sliding window at that
          position as the peak window.
        5) grow the window at the peak window position toward the top of the column and toward the bottom of the column
         such that the resulting gradient does not changes more than self.PERCENT_UNIFORMITY_THRESHOLD.
          Record the new width and locations (top, bottom) of the inflated peak window.
        """

        windows = list()
        grads = list()
        covs = list()
        compressed_id_to_row_id = dict()
        row_id_to_compressed_id = dict()

        for i in range(len(vertical_gradient_col)):  # TODO: subtract int(self.SLIDING_WINDOW_SIZE) from the range
            window = (i, i + int(self.SLIDING_WINDOW_SIZE))

            if self.window_is_excluded(windows_to_be_excluded, window):
                continue

            grad, cov = self.find_covering_non_negative_gradient_and_coverage(
                background_vert_gradient_col=vertical_gradient_col, window=window)
            windows.append(window)
            grads.append(grad)
            covs.append(cov)
            compressed_id_to_row_id[len(windows) - 1] = i
            row_id_to_compressed_id[i] = len(windows) - 1

        max_grad_index = int(np.argmax(grads))
        grad = grads[max_grad_index]
        cov = covs[max_grad_index]
        idx_top = compressed_id_to_row_id[max_grad_index]

        # if we find that the covering gradient at peak window position is negative
        # do not tune further the window width and return immediately the negative gradient value which
        # indicates failure
        if grad <= 0:
            return grad, cov, idx_top, idx_top + int(self.SLIDING_WINDOW_SIZE)

        # grow the region with the max grad with width int(self.SLIDING_WINDOW_SIZE) both toward the top and
        # toward the bottom until the resulting gradients do not differ more than self.PERCENT_UNIFORMITY_THRESHOLD
        for idx in reversed(range(idx_top)):
            window = (idx, idx_top + int(self.SLIDING_WINDOW_SIZE))
            if self.window_is_excluded(windows_to_be_excluded, window):
                idx_top = idx + 1
                break

            if (abs(grad - grads[row_id_to_compressed_id[idx]]) / grad * 100.0 >=
                    self.PERCENT_UNIFORMITY_THRESHOLD):
                # TODO: consider a weighted sum of the counts and the gradient levels instead of just the gradients
                idx_top = idx + 1
                break

        idx_bottom = idx_top + int(self.SLIDING_WINDOW_SIZE)

        for idx in range(idx_bottom, len(vertical_gradient_col)):
            window = (idx_top, idx)
            if self.window_is_excluded(windows_to_be_excluded, window):
                idx_bottom = idx - 1
                break
            # window (idx_top, idx) not being excluded implies that the window
            # (idx - int(self.SLIDING_WINDOW_SIZE), idx) will not be excluded as well
            if (abs(grad - grads[row_id_to_compressed_id[idx - int(self.SLIDING_WINDOW_SIZE)]]) / grad * 100.0 >=
                    self.PERCENT_UNIFORMITY_THRESHOLD):
                # TODO: consider a weighted sum of the counts and the gradient levels instead of just the gradients
                idx_bottom = idx - 1
                break

        return grad, cov, idx_top, idx_bottom

    def compute_size_of_shadow_region(self, vert_grad_col, window) -> tuple[bool, int, int]:
        """
        Helper method to determine the size of the shadow region in a more precise way.
        :param vert_grad_col: the vertical gradient component for a specific column
        :param window: the top and bottom indices of the region of interest
        :return: a boolean indicating if this is a shadow region and
          the top, the bottom and the range of the visible shadow region
          is_shadow_region (bool) - a boolean indicating if this is a shadow region
          top (int) - the top index of the visible shadow region
          bottom (int) - the bottom index of the visible shadow region
        """

        """
        The Algorithm:

        The input is the grown peak window which is the output of the method
        `find_max_covering_intensity_gradient_and_coverage`.

        The grown peak window needs to be grown from the top and adjusted from the bottom in order to
        include the negative intensity gradient subregion of the shadow region. In case a subregion  with
        negative intensity gradient is not found on the top of the positive gradient subregion then there is no shadow
        region covering the positive gradient subregion supplied by `find_max_covering_intensity_gradient_and_coverage`.
        In such case return False indicating we have not found shadow region in the indicated position.
        All of this functionality is implemented in the code below.

        An Example:
        For example, let us take a look at col 860 and refer to
         [Figure 1](https://nike.box.com/s/4chjx06oukuf8n7b7ydqkuu9c87nej9y) which depicts the gradient of
         the grayscale intensity for this column. The ellipse in red delineates the shadow region intersected
         with column 860. The positive gradient subregion of the shadow region is already captured (albeit imprecisely)
         in the output of `find_max_covering_intensity_gradient_and_coverage`. The goal is to grow this subregion so
         that it will eventually cover the negative gradient subregion of the shadow region.
        After the algorithm below executes the grown peak window will be adjusted to the top and to the bottom and for
        Column 860 of this example its width is delineated by blue line with width 272 on
        [Figure 2](https://nike.box.com/s/hq6xjxt58llt84623dek5ldk64nzz3kt).

        The Details:
        1) Compute the width of the grown peak window, as well as the exploration interval from the top
        `above_threshold` and the adjustment interval from the bottom `below_threshold`.

        2) Compute the partial sums of the intensity gradient values for each sub-interval with top index within
         the interval `above_threshold`. These partial sums are stored in the variable `sums_above`.

        3) Find the partial sum with minimum value for the intensity gradient among all partial sums and store it in
         `min_grad_sum`. This partial sum corresponds to the covering interval on the top for the sub-region with
         the negative intensity gradients.

        4) If the partial sum with minimum value for the intensity gradient is greater or equal to 0 then
        there is no negative intensity gradient subregion immediately on the top of the supplied positive gradient
        subregion. In such case return False.

        5) Now that we have determined the negative intensity gradient subregion we would like to adjust additionally
        the positive intensity gradient subregion by finding the precise width of the latter which minimizes the
        difference of the negative gradient area and the area of the positive gradient.

        6) return the new position of the adjusted positive intensity gradient subregion and estimated negative
        intensity gradient subregion.
        """

        width = window[1] - window[0]
        # assume the negative gradient portion of the shadow region is
        # immediately on the top of the max gradient region
        idx_bottom = window[1]
        # assume that the width of the negative gradient portion of the shadow region is the same as the width
        # of the positive gradient portion of the shadow region
        idx_top = window[0]

        if width <= 0:
            return False, idx_top, idx_bottom

        above_threshold = idx_top - max(0, idx_top - int(self.REGION_EXPLORATION_FACTOR_ABOVE) * width)
        below_threshold = (min(len(vert_grad_col), idx_bottom + int(self.REGION_EXPLORATION_FACTOR_BELOW) * width)
                           - idx_bottom)

        sums_above = list()
        for i in range(0, above_threshold):
            sums_above.append(np.sum(vert_grad_col[idx_top - i: idx_top]))

        # find the max cover sum which is closest to 0
        min_grad_idx = np.argmin(sums_above)
        min_grad_sum = sums_above[min_grad_idx]

        if min_grad_sum >= 0:
            return False, idx_top, idx_bottom

        sums_list = list()
        for i in range(0, below_threshold):
            sums_list.append(np.sum(vert_grad_col[idx_top - min_grad_idx: idx_bottom + i]))

        sums = np.array(sums_list)
        sums_non_neg = sums[np.where(sums >= 0)]

        if len(sums_non_neg) > 0:
            min_non_neg_idx = np.argmin(sums_non_neg)
            min_non_neg_sum = sums_non_neg[min_non_neg_idx]
            bottom_grad_idx = np.where(sums == min_non_neg_sum)[0][0]
        else:
            return False, idx_top, idx_bottom

        idx_top_new = idx_top - min_grad_idx
        idx_bottom_new = idx_bottom + bottom_grad_idx

        return True, idx_top_new, idx_bottom_new

    def estimate_visible_shadow_size(self, vert_intensity_col: NDArray[np.uint8], idx_top: int,
                                     idx_bottom: int, window_width: int) -> tuple[int, int, int, int]:
        """
        Estimate the visible shadow size
        :param vert_intensity_col: the vertical intensity component for a specific column
        :param idx_top: the top index of the initially estimated shadow region
        :param idx_bottom: the bottom index of the initially estimated shadow region
        :param window_width: the width of the window obtained from find_max_covering_intensity_gradient_and_coverage
        :return: the visible shadow top and bottom indices, the index of min intensity (thickest shadow), the value of
            min intensity level
        """

        """
        The Algorithm:

        The input is the output of the method `compute_size_of_shadow_region`.
        The goal of this method is to find the visible shadow region which is a subregion of the initially estimated
        shadow region by the method `compute_size_of_shadow_region`.
        After the algorithm below executes the initially estimated width for the shadow region computed by
          `compute_size_of_shadow_region` will be reduced to the visible shadow region width.
        This is achieved by computing the 0.45 quantile of the difference of the intensity gradient values between
        the initially estimated shadow region and the interpolated shadowless baseline.

        An Example:
        Column 860 of DQ6013-010_391629087_D_E_4X5
        The initially estimated width of the shadow computed with `compute_size_of_shadow_region` is delineated
        by blue line with width 272 on
        [Figure 1](https://nike.box.com/s/hq6xjxt58llt84623dek5ldk64nzz3kt).
        After this method executes the visible shadow region is delineated by the blue line with width 88.
        """

        # interpolate the intensity between idx_top - min_grad_idx and idx_bottom + bottom_grad_idx
        # to find the exact position of the shadow region
        idx_interp_above = (max(0, idx_top - int(self.REGION_INTERPOLATION_FACTOR_ABOVE) * window_width))

        idx_interp_below = (
            min(len(vert_intensity_col), idx_bottom + int(self.REGION_INTERPOLATION_FACTOR_BELOW) * window_width))

        x1 = np.arange(idx_interp_above, idx_top, 1, dtype=int)
        x2 = np.arange(idx_bottom, idx_interp_below, 1, dtype=int)
        x = np.concatenate((x1, x2))

        y1 = vert_intensity_col[idx_interp_above: idx_top]
        y2 = vert_intensity_col[idx_bottom: idx_interp_below]
        y = np.concatenate((y1, y2)).astype(float)

        f = interp1d(x, y, kind='slinear')

        x_new = np.linspace(idx_top, idx_bottom, idx_bottom - idx_top)
        grayscale_intensity_shadow_removed = f(x_new)

        grayscale_intensity_with_shadow = vert_intensity_col[idx_top: idx_bottom]
        diff_intensity = grayscale_intensity_shadow_removed - grayscale_intensity_with_shadow

        max_idx = np.argmax(diff_intensity)
        max_diff_intensity = diff_intensity.max()
        min_diff_intensity = diff_intensity.min()
        threshold = (max_diff_intensity - min_diff_intensity) * self.VISIBLE_SHADOW_QUANTILE

        delta_top_visible_shadow = np.argmax(diff_intensity > threshold)
        visible_shadow = diff_intensity[diff_intensity > threshold]

        visible_shadow_width = len(visible_shadow)
        delta_bottom_visible_shadow = delta_top_visible_shadow + visible_shadow_width

        return (idx_top + delta_top_visible_shadow, idx_top + delta_bottom_visible_shadow,
                idx_top + max_idx, max_diff_intensity)


def get_background_height_and_width(img: NDArray[np.uint8], loc: Location, dimension: int) -> tuple[int, int]:
    """
    Get the height and width of the background area
    :param img: np.ndarray, image to outpaint
    :param loc: Location, location of the outpaint
    :param dimension: int, dimension of the outpaint
    :return: tuple with height and width of the background area
    """
    if loc == Location.LEFT or loc == Location.RIGHT:
        return img.shape[0], dimension
    elif loc == Location.TOP or loc == Location.BOTTOM:
        return dimension, img.shape[1]


def get_interpolation_dataset(
        img: NDArray[np.uint8], background_height: int, background_width: int,
        color_chan_index: int, location: Location) -> NDArray[np.uint8]:
    """
    Get the dataset for the interpolation
    :param img: np.ndarray, image to outpaint
    :param background_height: int, height of the background area
    :param background_width: int, width of the background area
    :param color_chan_index: int, index of the color channel
    :param location: Location, location of the outpaint
    return interpolation dataset for single color channel (np.ndarray)
    """
    if location == Location.LEFT:
        return img[0:background_height, 0:background_width, color_chan_index]
    elif location == Location.RIGHT:
        return np.fliplr(img[0:background_height, -background_width:, color_chan_index])
    elif location == Location.TOP:
        return img[0:background_height, 0:background_width, color_chan_index]
    elif location == Location.BOTTOM:
        return np.flipud(img[-background_height:, 0:background_width, color_chan_index])


class ShadowRemover:
    def __init__(self):
        self.config = OutpainterConfig().config
        self.BLUR_KERNEL_SIZE = tuple(
            [self.config["shadow_remover"]["blur"]["parameters"]["size"]] * 2
        )
        self.BLUR_SIGMA = self.config["shadow_remover"]["blur"]["parameters"]["sigma"]
        self.DROPOUT_COLUMNS_INCREMENT_STEP_TOP = (
            self.config)["shadow_remover"]["parameters"]["dropout_columns"]["top"]["increment_step"]
        self.DROPOUT_COLUMNS_INCREMENT_STEP_BOTTOM = (
            self.config)["shadow_remover"]["parameters"]["dropout_columns"]["bottom"]["increment_step"]
        self.DROPOUT_COLUMNS_BACKGROUND_SIZE_TOP = (
            self.config)["shadow_remover"]["parameters"]["dropout_columns"]["top"]["background_size"]
        self.DROPOUT_COLUMNS_BACKGROUND_SIZE_BOTTOM = (
            self.config)["shadow_remover"]["parameters"]["dropout_columns"]["bottom"]["background_size"]

        self._shadowDetector = ShadowDetector(add_only_shadow_regions=True)

    @staticmethod
    def interpolate_feature(col_indices: NDArray[int], col_indices_interp: NDArray[int],
                            y_data: NDArray[Any]) -> NDArray[Any]:
        """
        Interpolate the features of the computed shadow regions
        :param col_indices: list of column indices
        :param col_indices_interp: array of continuous column indices needed for the interpolation
        :param y_data: list of feature values to interpolate
        """
        y_data_interpolated = interp1d(col_indices, y_data, bounds_error=False, kind='cubic')
        y_data_interp_arr = y_data_interpolated(col_indices_interp)

        return y_data_interp_arr

    def get_shadow_region_features(self, shadow_regions: List[VerticalGradientRegion],
                                   interpolate_missing_regions, sort_by_index: bool = False) -> ShadowRegionsFeatures:
        """
        Interpolate the shadow regions
        :param shadow_regions: list of shadow regions
        :param interpolate_missing_regions: bool, whether to interpolate the missing regions
        :param sort_by_index: whether to sort the shadow regions by index
        :return: list of shadow regions with interpolated regions
        """
        if sort_by_index:
            shadow_regions.sort(key=lambda x: x.col_idx)

        idx_top_list = []
        idx_top_viz_list = []
        idx_darkest_list = []
        idx_bottom_list = []
        idx_bottom_viz_list = []
        intensity_darkest_list = []
        intensity_viz_thresh_list = []
        idx_col_list = []

        need_to_interpolate = False
        cur_idx = 0
        next_idx = 1
        while cur_idx < len(shadow_regions):
            cur_region = shadow_regions[cur_idx]
            if next_idx < len(shadow_regions):
                next_region = shadow_regions[next_idx]
            else:
                next_region = None

            if next_region is not None and cur_region.col_idx + 1 < next_region.col_idx:
                # there is a gap between the regions which we need to interpolate
                need_to_interpolate = True

            next_idx += 1
            cur_idx += 1

            idx_col_list.append(cur_region.col_idx)
            idx_top_list.append(cur_region.row_idx_start)
            idx_top_viz_list.append(cur_region.row_idx_visible_start)
            idx_darkest_list.append(cur_region.row_idx_darkest)
            idx_bottom_viz_list.append(cur_region.row_idx_visible_end)
            idx_bottom_list.append(cur_region.row_idx_end)
            intensity_darkest_list.append(cur_region.darkest_intensity_delta)
            intensity_viz_thresh_list.append(cur_region.darkest_intensity_delta * 0.45)

        if need_to_interpolate and interpolate_missing_regions:
            col_indices_interp_arr = np.arange(idx_col_list[0], idx_top_list[-1]+1)
            col_indices_arr = np.array(idx_col_list)
            top_indices_arr = np.array(idx_top_list)
            top_viz_indices_arr = np.array(idx_top_viz_list)
            darkest_indices_arr = np.array(idx_darkest_list)
            bottom_viz_indices_arr = np.array(idx_bottom_viz_list)
            bottom_indices_arr = np.array(idx_bottom_list)
            intensity_darkest_arr = np.array(intensity_darkest_list)
            intensity_viz_thresh_arr = np.array(intensity_viz_thresh_list)

            idx_top_interp_arr = ShadowRemover.interpolate_feature(
                col_indices_arr, col_indices_interp_arr, top_indices_arr)

            idx_top_viz_interp_arr = ShadowRemover.interpolate_feature(
                col_indices_arr, col_indices_interp_arr, top_viz_indices_arr)

            idx_darkest_interp_arr = ShadowRemover.interpolate_feature(
                col_indices_arr, col_indices_interp_arr, darkest_indices_arr)

            idx_bottom_viz_interp_arr = ShadowRemover.interpolate_feature(
                col_indices_arr, col_indices_interp_arr, bottom_viz_indices_arr)

            idx_bottom_interp_arr = ShadowRemover.interpolate_feature(
                col_indices_arr, col_indices_interp_arr, bottom_indices_arr)

            intensity_darkest_interp_arr = ShadowRemover.interpolate_feature(
                col_indices_arr, col_indices_interp_arr, intensity_darkest_arr)

            intensity_viz_thresh_interp_arr = ShadowRemover.interpolate_feature(
                col_indices_arr, col_indices_interp_arr, intensity_viz_thresh_arr)

            return ShadowRegionsFeatures(col_indices_interp_arr,
                                         idx_top_interp_arr, idx_top_viz_interp_arr,
                                         idx_darkest_interp_arr, idx_bottom_viz_interp_arr, idx_bottom_interp_arr,
                                         intensity_darkest_interp_arr, intensity_viz_thresh_interp_arr)
        else:
            return ShadowRegionsFeatures(np.array(idx_col_list),
                                         np.array(idx_top_list), np.array(idx_top_viz_list),
                                         np.array(idx_darkest_list), np.array(idx_bottom_viz_list),
                                         np.array(idx_bottom_list), np.array(intensity_darkest_list),
                                         np.array(intensity_viz_thresh_list))

    def reduce_training_dataset_columns(self, gradient_background_size: int) -> range:
        """
        Reduce the training dataset columns using the top and bottom increment steps for the corresponding
         background sizes. Interpolate everything in between.
        :param gradient_background_size: int, size of the gradient background
        :return: range of the columns to be used for training
        """
        if gradient_background_size <= 0:
            return range(0, 0)

        top_background_size = self.DROPOUT_COLUMNS_BACKGROUND_SIZE_TOP
        bottom_background_size = self.DROPOUT_COLUMNS_BACKGROUND_SIZE_BOTTOM
        top_increment_step = self.DROPOUT_COLUMNS_INCREMENT_STEP_TOP
        bottom_increment_step = self.DROPOUT_COLUMNS_INCREMENT_STEP_BOTTOM
        if gradient_background_size >= top_background_size:
            increment_step = top_increment_step
        elif top_background_size > gradient_background_size > bottom_background_size:
            increment_step = (
                int(np.ceil(bottom_increment_step +
                            (top_increment_step - bottom_increment_step) /
                            (top_background_size - bottom_background_size) *
                            (gradient_background_size - bottom_background_size))))
        else:
            increment_step = bottom_increment_step

        return range(0, gradient_background_size, increment_step)

    def detect_shadow(self, img: NDArray[np.uint8], gradient_backgrounds: dict[Location, int]) ->\
            dict[Location, dict[int, List[VerticalGradientRegion]]]:
        """
        Detect the shadow regions in the image
        :param img: image to detect the shadow regions in
        :param gradient_backgrounds: dict with gradient backgrounds
        :return: list of shadow regions
        """

        shadow_regions = dict()
        for location, gradient_background_size in gradient_backgrounds.items():

            # for now we limit the shadow detection, removal and extrapolation only on left and right side outpaints
            if location is not Location.LEFT:  # TODO: and location is not Location.RIGHT:
                continue

            if not gradient_background_size > 0:
                continue

            background_column_range = self.reduce_training_dataset_columns(gradient_background_size)
            background_height, background_width = get_background_height_and_width(
                img, location, gradient_background_size)

            shadow_regions_for_cur_location = dict()
            for color_channel_index in range(0, 3):

                gradient_background = (
                    get_interpolation_dataset(img, background_height, background_width, color_channel_index, location))

                shadow_regions_for_cur_location[color_channel_index] = (
                    self._shadowDetector.find_vertical_gradient_regions(
                        gradient_background, list(background_column_range)))

            shadow_regions[location] = shadow_regions_for_cur_location

        return shadow_regions

    def remove_shadow(self, gradient_background: NDArray[np.uint8], features: ShadowRegionsFeatures) ->\
            NDArray[np.uint8]:
        # TODO: implement the shadow removal for specific location and specific color channel
        pass

    def remove_shadows_all_locations(self, img: NDArray[np.uint8], gradient_backgrounds: dict[Location, int],
                                     shadow_regions_all_locations: dict[Location, dict[int, List[VerticalGradientRegion]]],
                                     interpolate_missing_regions=True) -> NDArray[np.uint8]:
        """
        :param img: image to remove the shadow from
        :param gradient_backgrounds: dict with gradient background dimensions
        :param shadow_regions_all_locations: list of shadow regions for all background gradient locations
        :param interpolate_missing_regions: bool, whether to interpolate the missing regions
        :return: image with removed shadow
        """

        for location, shadow_regions_per_location in shadow_regions_all_locations.items():
            gradient_background_dimension = gradient_backgrounds[location]
            background_height, background_width = get_background_height_and_width(
                img, location, gradient_background_dimension)

            for color_channel_index in range(0, 3):
                shadow_regions = shadow_regions_per_location[color_channel_index]
                features = self.get_shadow_region_features(
                    shadow_regions, interpolate_missing_regions, sort_by_index=False)

                assert features
                gradient_background = (
                    get_interpolation_dataset(img, background_height, background_width, color_channel_index, location))
                assert gradient_background

                # idx_col_arr
                # idx_top_arr
                # idx_bottom_arr
        return img


class ShadowExtrapolator:
    def __init__(self):
        self.config = OutpainterConfig().config

    def extrapolate_shadow(self, img: NDArray[np.uint8], shadow_regions: List[VerticalGradientRegion]) -> NDArray[np.uint8]:
        """
        Extrapolate the shadow regions in the image
        :param img: image to extrapolate the shadow regions in
        :param shadow_regions: list of shadow regions
        :return: image with extrapolated shadow regions
        """
        for shadow_region in shadow_regions:
            img = self.extrapolate_shadow_region(img, shadow_region)
        return img


class Outpainter:
    def __init__(self):
        self.config = OutpainterConfig().config
        self.NOISELESS_OUTPAINT_BLUR_KERNEL_SIZE = tuple(
            [self.config["outpainter"]["noiseless_outpaint_blur"]["parameters"]["size"]] * 2
        )
        self.NOISELESS_OUTPAINT_BLUR_SIGMA = self.config["outpainter"]["noiseless_outpaint_blur"]["parameters"]["sigma"]
        self.NOISE_ESTIMATE_BLUR_KERNEL_SIZE = tuple(
            [self.config["outpainter"]["noise_estimate_blur"]["parameters"]["size"]] * 2
        )
        self.NOISE_ESTIMATE_BLUR_SIGMA = self.config["outpainter"]["noise_estimate_blur"]["parameters"]["sigma"]

        self.INTERPOLATION_METHOD = self.config["outpainter"]["parameters"]["interpolation_method"]
        self.SAMPLE_COLUMNS_COUNT = self.config["outpainter"]["parameters"]["sample_columns_count"]
        self.SAMPLE_ROWS_COUNT = self.config["outpainter"]["parameters"]["sample_rows_count"]
        self.NOISE_MODEL = self.config["outpainter"]["parameters"]["noise_model"]
        self.NOISE_CONSTRUCTION_METHOD = self.config["outpainter"]["parameters"]["noise_construction_method"]
        self.NOISE_INJECTION_METHOD = self.config["outpainter"]["parameters"]["noise_injection_method"]

        # colored noise model specific parameters
        self.COLORED_NOISE_BLOCK_SIZE = self.config["outpainter"]["parameters"]["colored_noise_block_size"]
        self.STANDARD_DEVIATION_FACTOR = self.config["outpainter"]["parameters"]["standard_deviation_factor"]
        self.COLORED_NOISE_BLUR_KERNEL_SIZE = tuple(
            [self.config["outpainter"]["colored_noise_blur"]["parameters"]["size"]] * 2
        )
        self.COLORED_NOISE_BLUR_SIGMA = self.config["outpainter"]["colored_noise_blur"]["parameters"]["sigma"]
        self.APPLY_BLUR_ON_PAINT = self.config["outpainter"]["parameters"]["apply_blur_on_paint"]

        self.COLUMN_SELECTION_METHOD = self.config["outpainter"]["parameters"]["column_selection_method"]
        self.NUMER_OF_COLUMNS_TO_CONSIDER = self.config["outpainter"]["parameters"]["number_of_columns_to_consider"]

        self.ROW_SELECTION_METHOD = self.config["outpainter"]["parameters"]["row_selection_method"]
        self.NUMER_OF_ROWS_TO_CONSIDER = self.config["outpainter"]["parameters"]["number_of_rows_to_consider"]

        self._img_blur_noiseless_outpaint = None
        self._img_blur_noise_estimate = None
        self._img_gray_noise_estimate = None
        self._img_blur_gray_noise_estimate = None

    def paint(
        self,
        img: NDArray[np.uint8],
        background_bbox: dict[Location, int],
        outpaint_bbox: dict[Location, int],
    ) -> NDArray[np.uint8]:
        """
        Extend the image by outpainting the specified areas in the outpaint box
        :param img: image to outpaint
        :param background_bbox: dict with padding boundary box information
        :param outpaint_bbox: dict with outpainting boundary box information
        :return: image with outpainted areas
        """

        image_height, image_width = img.shape[:2]

        top_background_height = background_bbox.get(Location.TOP, 0)
        top_outpaint_height = outpaint_bbox.get(Location.TOP, 0)
        top_outpaint_width = image_width if Location.TOP in outpaint_bbox else 0

        bottom_background_height = background_bbox.get(Location.BOTTOM, 0)
        bottom_outpaint_height = outpaint_bbox.get(Location.BOTTOM, 0)
        bottom_outpaint_width = image_width if Location.BOTTOM in outpaint_bbox else 0

        left_background_width = background_bbox.get(Location.LEFT, 0)
        left_outpaint_width = outpaint_bbox.get(Location.LEFT, 0)
        left_outpaint_height = image_height if Location.LEFT in outpaint_bbox else 0

        right_background_width = background_bbox.get(Location.RIGHT, 0)
        right_outpaint_width = outpaint_bbox.get(Location.RIGHT, 0)
        right_outpaint_height = image_height if Location.RIGHT in outpaint_bbox else 0

        self._img_blur_noiseless_outpaint = cv2.GaussianBlur(
            img, self.NOISELESS_OUTPAINT_BLUR_KERNEL_SIZE, self.NOISELESS_OUTPAINT_BLUR_SIGMA)
        if self.NOISELESS_OUTPAINT_BLUR_KERNEL_SIZE != self.NOISE_ESTIMATE_BLUR_KERNEL_SIZE or \
                self.NOISELESS_OUTPAINT_BLUR_SIGMA != self.NOISE_ESTIMATE_BLUR_SIGMA:
            self._img_blur_noise_estimate = cv2.GaussianBlur(
                img, self.NOISE_ESTIMATE_BLUR_KERNEL_SIZE, self.NOISE_ESTIMATE_BLUR_SIGMA)
        else:
            self._img_blur_noise_estimate = self._img_blur_noiseless_outpaint

        self._img_gray_noise_estimate = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self._img_blur_gray_noise_estimate = cv2.cvtColor(self._img_blur_noise_estimate, cv2.COLOR_BGR2GRAY)

        result = img
        # create left outpaint
        if left_outpaint_width > 0:
            x = np.arange(0, left_outpaint_height, 1, dtype=int)
            y = np.arange(
                left_outpaint_width, left_outpaint_width + left_background_width, 1, dtype=int
            )
            left_rgb_outpaint = self._extrapolate_rgb_background_horiz(
                x, y, img, image_height, left_background_width, left_outpaint_width, Location.LEFT)
            result = np.hstack((left_rgb_outpaint[:, 0:left_outpaint_width], result))

        # create right outpaint
        if right_outpaint_width > 0:
            x = np.arange(0, right_outpaint_height, 1, dtype=int)
            y = np.arange(
                right_outpaint_width, right_background_width + right_outpaint_width , 1, dtype=int
            )
            right_rgb_outpaint = self._extrapolate_rgb_background_horiz(
                x, y, img, image_height, right_background_width, right_outpaint_width, Location.RIGHT)
            result = np.hstack((result, right_rgb_outpaint))

        # create top outpaint
        if top_outpaint_height > 0:
            x = np.arange(top_outpaint_height, top_outpaint_height + top_background_height, 1, dtype=int)
            y = np.arange(0, top_outpaint_width, 1, dtype=int)
            top_rgb_outpaint = self._extrapolate_rgb_background_vert(
                x, y, img, image_width, top_background_height, top_outpaint_height, Location.TOP)

            result = np.vstack((top_rgb_outpaint, result))

        # create bottom outpaint
        if bottom_outpaint_height > 0:
            x = np.arange(bottom_outpaint_height, bottom_outpaint_height + bottom_background_height, 1, dtype=int)
            y = np.arange(0, bottom_outpaint_width, 1, dtype=int)
            bottom_rgb_outpaint = self._extrapolate_rgb_background_vert(
                x, y, img, image_width, bottom_background_height, bottom_outpaint_height, Location.BOTTOM)
            result = np.vstack((result, bottom_rgb_outpaint))

        return result.astype(np.uint8)

    def _add_white_noise_to_rgb_outpaint_via_color_channel_horiz(
            self,  x: NDArray[int], y: NDArray[int], img: NDArray[np.uint8],
            image_height: int, background_width: int, outpaint_width: int, location: Location) -> NDArray[np.uint8]:
        """
        Add noise to the RGB outpaint via color channel injection for horizontal outpaints (left, right outpaints)
        :param x (int): x coordinates of the data points
        :param y (int): y coordinates of the data points
        :param img: image to outpaint
        :param image_height: height of the image
        :param background_width: width of the background area
        :param outpaint_width: width of the area to outpaint
        :param location: location of the outpaint
        :return: RGB outpaint with noise
        """
        channel_outpaints = list()
        for i in range(0, 3):
            data = self._get_interpolation_dataset(self._img_blur_noiseless_outpaint,
                                                   image_height, background_width, i, location)

            # Create interpolator and use it to extrapolate the background data into the outpaint area
            channel_outpaint = self._extrapolate_color_channel(x, y, data, image_height, outpaint_width)

            background_noise = self._get_noise_dataset(self._img_blur_noise_estimate,
                                                       img, image_height, background_width, i, location)

            channel_outpaint_noise = self._generate_channel_white_noise_horiz(background_noise, image_height,
                                                                          background_width, outpaint_width)
            channel_outpaints.append(np.add(channel_outpaint, channel_outpaint_noise).astype(np.uint8))

        rgb_outpaint_noisy = np.dstack(tuple(channel_outpaints))
        return rgb_outpaint_noisy

    def _get_outpaint_gaussian_colored_noise_channel_horiz(
            self, background_noise: NDArray[np.uint8], image_height: int, background_width: int,
            outpaint_width: int) -> NDArray[np.uint8]:
        """
        Get the outpaint region with colored Gaussian noise
        :param background_noise: np.ndarray, background noise
        :param image_height (int): height of the image
        :param background_width (int): width of the background area
        :param outpaint_width (int): width of the area to outpaint
        return outpaint region with colored noise (np.ndarray)
        """
        def get_outpaint_slice():
            """
            a helper function to generate a slice of the outpaint region with colored noise
            Note: a slice is a sequences of blocks containing noise with the size of the colored noise block
            """
            # number of blocks which can be stacked horizontally in the outpaint region
            number_of_blocks_horiz_outpaint = outpaint_width // self.COLORED_NOISE_BLOCK_SIZE
            # leftover pixels less than one block size of width in the outpaint region
            remainder_horiz_outpaint = outpaint_width % self.COLORED_NOISE_BLOCK_SIZE

            grayscale_white_noise = (
                np.random.normal(0, 1, (row_count, number_of_blocks_horiz_outpaint)).astype(int))
            horiz_slice = (
                np.real(np.dot(cov_sqrt_gray, grayscale_white_noise)).reshape(self.COLORED_NOISE_BLOCK_SIZE, -1))

            if remainder_horiz_outpaint > 0:
                grayscale_white_noise = np.random.normal(0, 1, (row_count, 1)).astype(int)
                outpaint_grayscale_noise_block = np.real(np.dot(cov_sqrt_gray, grayscale_white_noise)).reshape(
                    self.COLORED_NOISE_BLOCK_SIZE, -1)
                horiz_slice = np.hstack((horiz_slice, outpaint_grayscale_noise_block[:, 0:remainder_horiz_outpaint]))
            return horiz_slice

        # number of blocks which can be stacked horizontally in the gradient background region
        number_of_blocks_horiz = background_width // self.COLORED_NOISE_BLOCK_SIZE

        # number of blocks which can be stacked vertically in the outpaint region
        number_of_blocks_vert = image_height // self.COLORED_NOISE_BLOCK_SIZE
        # leftover pixels less than one block size of height in the gradient background region
        remainder_vert = image_height % self.COLORED_NOISE_BLOCK_SIZE

        row_count = self.COLORED_NOISE_BLOCK_SIZE * self.COLORED_NOISE_BLOCK_SIZE

        # gather all available blocks in all slices in the gradient background region noise as a training dataset
        # Note: a slice is a sequences of blocks containing noise with the size of self.COLORED_NOISE_BLOCK_SIZE
        slice_array = None
        for i in range(0, number_of_blocks_vert):
            background_noise_slice = background_noise[
                                     i * self.COLORED_NOISE_BLOCK_SIZE: (i+1) * self.COLORED_NOISE_BLOCK_SIZE,
                                     :number_of_blocks_horiz * self.COLORED_NOISE_BLOCK_SIZE]

            if slice_array is not None:
                slice_array = np.hstack((slice_array, background_noise_slice.reshape(row_count, -1)))
            else:
                slice_array = background_noise_slice.reshape(row_count, -1)

        # introducing colored Gaussian noise as the dot product of the square root of the covariance matrix of the
        # noise in the training dataset and a white Gaussian noise with empirically computed mean and std dev.
        cov_matrix_gray_noise = np.cov(slice_array)
        cov_sqrt_gray = self.STANDARD_DEVIATION_FACTOR * linalg.sqrtm(cov_matrix_gray_noise)

        # construct all slices with colored noise in the outpaint region
        outpaint_grayscale_noise = None
        for i in range(0, number_of_blocks_vert):
            outpaint_slice = get_outpaint_slice()
            if outpaint_grayscale_noise is not None:
                outpaint_grayscale_noise = np.vstack((outpaint_grayscale_noise, outpaint_slice))
            else:
                outpaint_grayscale_noise = outpaint_slice
        if remainder_vert > 0:
            outpaint_slice = get_outpaint_slice()
            outpaint_grayscale_noise = np.vstack((outpaint_grayscale_noise, outpaint_slice[0:remainder_vert, :]))

        return outpaint_grayscale_noise

    def _add_colored_noise_to_rgb_outpaint_via_color_channel_horiz(
            self,  x: NDArray[int], y: NDArray[int], img: NDArray[np.uint8],
            image_height: int, background_width: int, outpaint_width: int, location: Location) -> NDArray[np.uint8]:
        """
        Add noise to the RGB outpaint via color channel injection for horizontal outpaints (left, right outpaints)
        :param x (int): x coordinates of the data points
        :param y (int): y coordinates of the data points
        :param img: image to outpaint
        :param image_height: height of the image
        :param background_width: width of the background area
        :param outpaint_width: width of the area to outpaint
        :param location: location of the outpainted region
        :return: RGB outpaint with noise
        """
        channel_outpaints = list()
        for i in range(0, 3):
            data = self._get_interpolation_dataset(self._img_blur_noiseless_outpaint, image_height,
                                                   background_width, i, location)

            # Create interpolator and use it to extrapolate the background data into the outpaint area
            channel_outpaint = self._extrapolate_color_channel(x, y, data, image_height, outpaint_width)

            background_noise = self._get_noise_dataset(self._img_blur_noise_estimate, img,
                                                       image_height, background_width, i, location)

            channel_outpaint_noise = self._get_outpaint_gaussian_colored_noise_channel_horiz(
                background_noise, image_height, background_width, outpaint_width)

            channel_outpaints.append(np.add(channel_outpaint, channel_outpaint_noise).astype(np.uint8))

        rgb_outpaint_noisy = np.dstack(tuple(channel_outpaints))

        if self.APPLY_BLUR_ON_PAINT:
            return cv2.GaussianBlur(rgb_outpaint_noisy, self.COLORED_NOISE_BLUR_KERNEL_SIZE, self.COLORED_NOISE_BLUR_SIGMA)
        else:
            return rgb_outpaint_noisy

    def _add_white_noise_to_rgb_outpaint_via_luminance_horiz(
            self,  x: NDArray[int], y: NDArray[int],
            image_height: int, background_width: int, outpaint_width: int,
            location: Location) -> NDArray[np.uint8]:
        """
        Add Gaussian white noise to the RGB outpaint via luminance-based injection for horizontal outpaints (left and right outpaints)
        :param x (int): x coordinates of the data points
        :param y (int): y coordinates of the data points
        :param img: image to outpaint
        :param image_height: height of the image
        :param background_width: width of the background area
        :param outpaint_width: width of the area to outpaint
        :param location: location of the outpaint
        :return: RGB outpaint with noise
        """
        channel_outpaints = list()
        for i in range(0, 3):
            data = self._get_interpolation_dataset(self._img_blur_noiseless_outpaint, image_height,
                                                   background_width, i, location)

            # Create interpolator and use it to extrapolate the background data into the outpaint area
            channel_outpaint = self._extrapolate_color_channel(x, y, data, image_height, outpaint_width)

            channel_outpaints.append(channel_outpaint)

        rgb_channel_outpaint_noiseless = np.dstack(tuple(channel_outpaints))
        rgb_outpaint_noisy = self._inject_luminance_white_noise_horiz(rgb_channel_outpaint_noiseless,
                                                                  image_height,
                                                                  background_width, outpaint_width, location)
        return rgb_outpaint_noisy

    def _add_colored_noise_to_rgb_outpaint_via_luminance_horiz(
            self,  x: NDArray[int], y: NDArray[int],
            image_height: int, background_width: int, outpaint_width: int,
            location: Location) -> NDArray[np.uint8]:
        """
        Add colored Gaussian noise to the RGB outpaint via luminance-based injection for horizontal outpaints (left and right outpaints)
        :param x (int): x coordinates of the data points
        :param y (int): y coordinates of the data points
        :param image_height: height of the image
        :param background_width: width of the background area
        :param outpaint_width: width of the area to outpaint
        :param location: location of the outpaint
        :return: RGB outpaint with noise
        """
        channel_outpaints = list()
        for i in range(0, 3):
            data = self._get_interpolation_dataset(self._img_blur_noiseless_outpaint, image_height,
                                                   background_width, i, location)

            # Create interpolator and use it to extrapolate the background data into the outpaint area
            channel_outpaint = self._extrapolate_color_channel(x, y, data, image_height, outpaint_width)

            channel_outpaints.append(channel_outpaint)

        rgb_channel_outpaint_noiseless = np.dstack(tuple(channel_outpaints))
        rgb_outpaint_noisy = self._inject_luminance_colored_noise_horiz(
            rgb_channel_outpaint_noiseless,
            image_height, background_width, outpaint_width, location)
        return rgb_outpaint_noisy

    def _create_rgb_outpaint_with_gradient_background_horiz(
            self, x: NDArray[int], y: NDArray[int], img_blur: NDArray[np.uint8],
            image_height: int, background_width: int, outpaint_width: int,
            location: Location) -> NDArray[np.uint8]:
        """
        Create the RGB outpaint with gradient background noise
        :param x (int): x coordinates of the data points
        :param y (int): y coordinates of the data points
        :param img_blur (np.ndarray): image with applied Gaussian blur
        :param image_height (int): height of the image
        :param background_width (int): width of the background area
        :param outpaint_width (int): width of the area to outpaint
        :param location (Location): location of the outpaint
        """
        # do not inject any noise in the outpaint area in this case
        channel_outpaints = list()
        for i in range(0, 3):
            data = self._get_interpolation_dataset(img_blur, image_height, background_width, i, location)

            # Create interpolator and use it to extrapolate the background data into the outpaint area
            channel_outpaint = self._extrapolate_color_channel(x, y, data, image_height, outpaint_width)
            channel_outpaints.append(channel_outpaint)

        rgb_outpaint_noiseless = np.dstack(tuple(channel_outpaints))

        background_noise = self._get_grayscale_noise_dataset(
            self._img_blur_gray_noise_estimate, self._img_gray_noise_estimate, image_height, background_width, location)

        if outpaint_width > background_width:
            repeat_factor = outpaint_width // background_width
            remainder = outpaint_width % background_width
            outpaint_grayscale_noise = np.tile(background_noise, (1, repeat_factor))
            if remainder > 0:
                outpaint_grayscale_noise = np.hstack((outpaint_grayscale_noise, background_noise[:, 0:remainder]))
        else:
            outpaint_grayscale_noise = background_noise[:, 0:outpaint_width]

        rgb_outpaint_noisy = rgb_outpaint_noiseless + np.dstack(tuple([outpaint_grayscale_noise] * 3)).astype(np.uint8)

        return rgb_outpaint_noisy

    def _create_rgb_outpaint_noiseless_horiz(
            self,  x: NDArray[int], y: NDArray[int], img_blur: NDArray[np.uint8],
            image_height: int, background_width: int, outpaint_width: int,
            location: Location) -> NDArray[np.uint8]:
        """
        Create the RGB outpaint without noise
        :param x (int): x coordinates of the data points
        :param y (int): y coordinates of the data points
        :param img_blur (np.ndarray): image with applied Gaussian blur
        :param image_height (int): height of the image
        :param background_width (int): width of the background area
        :param outpaint_width (int): width of the area to outpaint
        :param location (Location): location of the outpaint
        """
        # do not inject any noise in the outpaint area in this case
        channel_outpaints = list()
        for i in range(0, 3):
            data = self._get_interpolation_dataset(img_blur, image_height, background_width, i, location)

            # Create interpolator and use it to extrapolate the background data into the outpaint area
            channel_outpaint = self._extrapolate_color_channel(x, y, data, image_height, outpaint_width)
            channel_outpaints.append(channel_outpaint)

        rgb_outpaint_noiseless = np.dstack(tuple(channel_outpaints))

        return rgb_outpaint_noiseless

    def _extrapolate_rgb_background_with_gaussian_white_noise_horiz(
            self, x: NDArray[int], y: NDArray[int],
            img: NDArray[np.uint8],
            image_height: int, background_width: int, outpaint_width: int, location: Location) -> NDArray[np.uint8]:
        """
        Extrapolate the RGB data to outpaint the specified area extending vertically
        :param x (int): x coordinates of the data points
        :param y (int): y coordinates of the data points
        :param img (np.ndarray): image to outpaint
        :param image_height (int): height of the image
        :param background_width (int): width of the background area
        :param outpaint_width (int): width of the area to outpaint
        :param location (Location): location of the outpaint
        """
        if self.NOISE_CONSTRUCTION_METHOD == "color-channel-based":
            rgb_outpaint = self._add_white_noise_to_rgb_outpaint_via_color_channel_horiz(
                x, y, img, image_height, background_width, outpaint_width, location)

        elif self.NOISE_CONSTRUCTION_METHOD == "luminance-based":
            rgb_outpaint = self._add_white_noise_to_rgb_outpaint_via_luminance_horiz(
                x, y, image_height, background_width, outpaint_width, location)
        else:
            raise ValueError(f'Unknown noise construction method: {self.NOISE_CONSTRUCTION_METHOD}')

        return rgb_outpaint

    def _extrapolate_rgb_background_with_gaussian_colored_noise_horiz(
            self, x: NDArray[int], y: NDArray[int],
            img: NDArray[np.uint8],
            image_height: int, background_width: int, outpaint_width: int, location: Location) -> NDArray[np.uint8]:
        """
        Extrapolate the RGB data to outpaint the specified area extending vertically and add gaussian colored noise
        :param x (int): x coordinates of the data points
        :param y (int): y coordinates of the data points
        :param img (np.ndarray): image to outpaint
        :param image_height (int): height of the image
        :param background_width (int): width of the background area
        :param outpaint_width (int): width of the area to outpaint
        :param location (Location): location of the outpaint
        """
        if self.NOISE_CONSTRUCTION_METHOD == "color-channel-based":
            rgb_outpaint = self._add_colored_noise_to_rgb_outpaint_via_color_channel_horiz(
                x, y, img, image_height, background_width, outpaint_width, location)

        elif self.NOISE_CONSTRUCTION_METHOD == "luminance-based":
            rgb_outpaint = self._add_colored_noise_to_rgb_outpaint_via_luminance_horiz(
                x, y, image_height, background_width, outpaint_width, location)
        else:
            raise ValueError(f'Unknown noise construction method: {self.NOISE_CONSTRUCTION_METHOD}')

        return rgb_outpaint

    def _extrapolate_rgb_background_horiz(
            self, x: NDArray[int], y: NDArray[int],
            img: NDArray[np.uint8],
            image_height: int, background_width: int, outpaint_width: int, location: Location) -> NDArray[np.uint8]:
        """
        Extrapolate the RGB data to outpaint the specified area extending horizontally
        :param x (int): x coordinates of the data points
        :param y (int): y coordinates of the data points
        :param img (np.ndarray): image to outpaint
        :param image_height (int): height of the image
        :param background_width (int): width of the background area
        :param outpaint_width (int): width of the area to outpaint
        :param location (Location): location of the outpaint
        """
        if self.NOISE_MODEL == "gradient-background":
            rgb_outpaint = self._create_rgb_outpaint_with_gradient_background_horiz(
                x, y, self._img_blur_noiseless_outpaint, image_height, background_width, outpaint_width, location)
        elif self.NOISE_MODEL == "gaussian-white":
            rgb_outpaint = self._extrapolate_rgb_background_with_gaussian_white_noise_horiz(
                x, y, img, image_height, background_width, outpaint_width, location)
        elif self.NOISE_MODEL == "gaussian-colored":
            rgb_outpaint = self._extrapolate_rgb_background_with_gaussian_colored_noise_horiz(
                x, y, img, image_height, background_width, outpaint_width, location)
        elif self.NOISE_MODEL == "none":
            rgb_outpaint = self._create_rgb_outpaint_noiseless_horiz(
                x, y, self._img_blur_noiseless_outpaint, image_height, background_width, outpaint_width, location)
        else:
            raise ValueError(f'Unknown noise model: {self.NOISE_MODEL}')

        return rgb_outpaint

    def _add_white_noise_to_rgb_outpaint_via_color_channel_vert(
            self,  x: NDArray[int], y: NDArray[int], img: NDArray[np.uint8],
            image_width: int, background_height: int, outpaint_height: int, location: Location) -> NDArray[np.uint8]:
        """
        Add Gaussian white noise to the RGB outpaint via color channel injection for horizontal outpaints (left, right outpaints)
        :param x (int): x coordinates of the data points
        :param y (int): y coordinates of the data points
        :param img: image to outpaint
        :param image_width: the width of the image
        :param background_height: height of the background area
        :param outpaint_height: height of the area to outpaint
        :param location: location of the outpaint
        :return: RGB outpaint with noise
        """
        channel_outpaints = list()
        for i in range(0, 3):
            data = self._get_interpolation_dataset(self._img_blur_noiseless_outpaint, background_height,
                                                   image_width, i, location)

            # Create interpolator and use it to extrapolate the background data into the outpaint area
            channel_outpaint = self._extrapolate_color_channel(x, y, data, outpaint_height, image_width)

            background_noise = self._get_noise_dataset(self._img_blur_noise_estimate, img,
                                                       background_height, image_width, i, location)

            channel_outpaint_noise = self._generate_channel_white_noise_vert(
                background_noise, image_width, background_height, outpaint_height)
            channel_outpaints.append(np.add(channel_outpaint, channel_outpaint_noise).astype(np.uint8))

        rgb_outpaint_noisy = np.dstack(tuple(channel_outpaints))

        return rgb_outpaint_noisy

    def _add_colored_noise_to_rgb_outpaint_via_color_channel_vert(
            self,  x: NDArray[int], y: NDArray[int], img: NDArray[np.uint8],
            image_width: int, background_height: int, outpaint_height: int, location: Location) -> NDArray[np.uint8]:
        """
        Add colored Gaussian noise to the RGB outpaint via color channel injection for horizontal outpaints (left, right outpaints)
        :param x (int): x coordinates of the data points
        :param y (int): y coordinates of the data points
        :param img: image to outpaint
        :param image_width: the width of the image
        :param background_height: height of the background area
        :param outpaint_height: height of the area to outpaint
        :param location: location of the outpaint
        :return: RGB outpaint with noise
        """
        channel_outpaints = list()
        for i in range(0, 3):
            data = self._get_interpolation_dataset(self._img_blur_noiseless_outpaint, background_height,
                                                   image_width, i, location)

            # Create interpolator and use it to extrapolate the background data into the outpaint area
            channel_outpaint = self._extrapolate_color_channel(x, y, data, outpaint_height, image_width)

            background_noise = self._get_noise_dataset(self._img_blur_noise_estimate, img, background_height,
                                                       image_width, i, location)

            channel_outpaint_noise = self._get_outpaint_gaussian_colored_noise_channel_vert(
                background_noise, image_width, background_height, outpaint_height)

            channel_outpaints.append(np.add(channel_outpaint, channel_outpaint_noise).astype(np.uint8))

        rgb_outpaint_noisy = np.dstack(tuple(channel_outpaints))

        if self.APPLY_BLUR_ON_PAINT:
            return cv2.GaussianBlur(rgb_outpaint_noisy, self.COLORED_NOISE_BLUR_KERNEL_SIZE, self.COLORED_NOISE_BLUR_SIGMA)
        else:
            return rgb_outpaint_noisy

    def _get_outpaint_gaussian_colored_noise_channel_vert(self, background_noise: NDArray[np.uint8], image_width: int,
                                                      background_height: int, outpaint_height: int) -> NDArray[np.uint8]:
        """
        Get the outpaint region with colored Gaussian noise
        :param background_noise: np.ndarray, background noise
        :param image_width (int): width of the image
        :param background_height (int): height of the background area
        :param outpaint_height (int): height of the area to outpaint
        return outpaint region with colored noise (np.ndarray)
        """
        def get_outpaint_slice():
            """
            a helper function to generate a slice of the outpaint region with colored noise
            Note: a slice is a sequences of blocks containing noise with the size of the colored noise block
            """
            # number of blocks which can be stacked horizontally in the outpaint region
            number_of_blocks_vert_outpaint = outpaint_height // self.COLORED_NOISE_BLOCK_SIZE
            # leftover pixels less than one block size of width in the outpaint region
            remainder_vert_outpaint = outpaint_height % self.COLORED_NOISE_BLOCK_SIZE
            grayscale_white_noise = (
                np.random.normal(0, 1, (number_of_blocks_vert_outpaint, col_count)).astype(int))
            vert_slice = (
                np.real(np.dot(grayscale_white_noise, cov_sqrt_gray)).reshape(-1, self.COLORED_NOISE_BLOCK_SIZE))

            if remainder_vert_outpaint > 0:
                grayscale_white_noise = np.random.normal(0, 1, (1, col_count)).astype(int)

                outpaint_grayscale_noise_block = np.real(np.dot(grayscale_white_noise, cov_sqrt_gray)).reshape(
                    -1, self.COLORED_NOISE_BLOCK_SIZE)
                vert_slice = np.vstack((vert_slice, outpaint_grayscale_noise_block[0:remainder_vert_outpaint, :]))
            return vert_slice

        # number of blocks which can be stacked vertically in the gradient background region
        number_of_blocks_vert = background_height // self.COLORED_NOISE_BLOCK_SIZE
        # leftover pixels less than one block size of height in the gradient background region
        remainder_vert = background_height % self.COLORED_NOISE_BLOCK_SIZE

        # number of blocks which can be stacked horizontally in the outpaint region
        number_of_blocks_horiz = image_width // self.COLORED_NOISE_BLOCK_SIZE
        # leftover pixels less than one block size of width in the gradient background region
        remainder_horiz = image_width % self.COLORED_NOISE_BLOCK_SIZE

        col_count = self.COLORED_NOISE_BLOCK_SIZE * self.COLORED_NOISE_BLOCK_SIZE

        if number_of_blocks_vert == 0:
            # edge case in which the block size is larger than the background height
            # in this case we tile the gradient background vertically until it reaches the height of the block
            stack_count = self.COLORED_NOISE_BLOCK_SIZE // remainder_vert
            stack_remainder = self.COLORED_NOISE_BLOCK_SIZE % remainder_vert
            background_noise = np.tile(background_noise, (stack_count, 1))
            background_noise = np.vstack((background_noise, background_noise[0:stack_remainder, :]))
            number_of_blocks_vert = 1
            remainder_vert = 0

        # gather all available blocks in all slices in the gradient background region as a training dataset
        # Note: a slice is a sequences of blocks containing noise with the size of self.COLORED_NOISE_BLOCK_SIZE
        slice_array = None
        for i in range(0, number_of_blocks_horiz):
            background_noise_slice = background_noise[
                                     :number_of_blocks_vert * self.COLORED_NOISE_BLOCK_SIZE,
                                     i * self.COLORED_NOISE_BLOCK_SIZE: (i+1) * self.COLORED_NOISE_BLOCK_SIZE]

            if slice_array is not None:
                slice_array = np.vstack((slice_array, background_noise_slice.reshape(-1, col_count)))
            else:
                slice_array = background_noise_slice.reshape(-1, col_count)

        # introducing colored Gaussian noise as the dot product of the square root of the covariance matrix of the
        # noise in the training dataset and a white Gaussian noise with empirically computed mean and std dev.
        cov_matrix_gray_noise = np.cov(slice_array.T)
        cov_sqrt_gray = self.STANDARD_DEVIATION_FACTOR * linalg.sqrtm(cov_matrix_gray_noise)

        # construct all slices with colored noise in the outpaint region
        outpaint_grayscale_noise = None
        for i in range(0, number_of_blocks_horiz):
            outpaint_slice = get_outpaint_slice()
            if outpaint_grayscale_noise is not None:
                outpaint_grayscale_noise = np.hstack((outpaint_grayscale_noise, outpaint_slice))
            else:
                outpaint_grayscale_noise = outpaint_slice
        if remainder_horiz > 0:
            outpaint_slice = get_outpaint_slice()
            outpaint_grayscale_noise = np.hstack((outpaint_grayscale_noise, outpaint_slice[:, 0:remainder_horiz]))

        return outpaint_grayscale_noise

    def _add_white_noise_to_rgb_outpaint_via_luminance_vert(
            self,  x: NDArray[int], y: NDArray[int],
            image_width: int, background_height: int, outpaint_height: int,
            location: Location) -> NDArray[np.uint8]:
        """
        Add Gaussian white noise to the RGB outpaint via luminance-based injection for horizontal outpaints (left and right outpaints)
        :param x (int): x coordinates of the data points
        :param y (int): y coordinates of the data points
        :param image_height: height of the image
        :param background_width: width of the background area
        :param outpaint_width: width of the area to outpaint
        :param location: location of the outpaint
        :return: RGB outpaint with noise
        """
        channel_outpaints = list()
        for i in range(0, 3):
            data = self._get_interpolation_dataset(self._img_blur_noiseless_outpaint, background_height,
                                                   image_width, i, location)

            # Create interpolator and use it to extrapolate the background data into the outpaint area
            channel_outpaint = self._extrapolate_color_channel(x, y, data, outpaint_height, image_width)

            channel_outpaints.append(channel_outpaint)

        rgb_channel_outpaint_noiseless = np.dstack(tuple(channel_outpaints))
        rgb_outpaint_noisy = self._inject_luminance_white_noise_vert(
            rgb_channel_outpaint_noiseless, image_width, background_height, outpaint_height, location)
        return rgb_outpaint_noisy

    def _add_colored_noise_to_rgb_outpaint_via_luminance_vert(
            self,  x: NDArray[int], y: NDArray[int],
            image_width: int, background_height: int, outpaint_height: int,
            location: Location) -> NDArray[np.uint8]:
        """
        Add colored Gaussian noise to the RGB outpaint via luminance-based injection for horizontal outpaints (left and right outpaints)
        :param x (int): x coordinates of the data points
        :param y (int): y coordinates of the data points
        :param image_height: height of the image
        :param background_width: width of the background area
        :param outpaint_width: width of the area to outpaint
        :param location: location of the outpaint
        :return: RGB outpaint with noise
        """
        channel_outpaints = list()
        for i in range(0, 3):
            data = self._get_interpolation_dataset(self._img_blur_noiseless_outpaint, background_height,
                                                   image_width, i, location)

            # Create interpolator and use it to extrapolate the background data into the outpaint area
            channel_outpaint = self._extrapolate_color_channel(x, y, data, outpaint_height, image_width)

            channel_outpaints.append(channel_outpaint)

        rgb_channel_outpaint_noiseless = np.dstack(tuple(channel_outpaints))
        rgb_outpaint_noisy = self._inject_luminance_colored_noise_vert(
            rgb_channel_outpaint_noiseless, image_width, background_height, outpaint_height, location)
        return rgb_outpaint_noisy

    def _create_rgb_outpaint_with_gradient_background_noise_vert(
            self, x: NDArray[int], y: NDArray[int], img_blur: NDArray[np.uint8],
            image_width: int, background_height: int, outpaint_height: int,
            location: Location) -> NDArray[np.uint8]:
        """
        Create the RGB outpaint with gradient background noise
        :param x (int): x coordinates of the data points
        :param y (int): y coordinates of the data points
        :param img_blur (np.ndarray): image with applied Gaussian blur
        :param image_width (int): width of the image
        :param background_height (int): height of the background area
        :param outpaint_height (int): height of the area to outpaint
        :param location (Location): location of the outpaint
        """
        # do not inject any noise in the outpaint area in this case
        channel_outpaints = list()
        for i in range(0, 3):
            data = self._get_interpolation_dataset(img_blur, background_height, image_width, i, location)

            # Create interpolator and use it to extrapolate the background data into the outpaint area
            channel_outpaint = self._extrapolate_color_channel(x, y, data, outpaint_height, image_width)
            channel_outpaints.append(channel_outpaint)

        rgb_outpaint_noiseless = np.dstack(tuple(channel_outpaints))

        background_noise = self._get_grayscale_noise_dataset(
            self._img_blur_gray_noise_estimate, self._img_gray_noise_estimate, background_height, image_width, location)

        if outpaint_height > background_height:
            repeat_factor = outpaint_height // background_height
            remainder = outpaint_height % background_height
            outpaint_grayscale_noise = np.tile(background_noise, (repeat_factor, 1))
            if remainder > 0:
                outpaint_grayscale_noise = np.vstack((outpaint_grayscale_noise, background_noise[0:remainder, :]))
        else:
            outpaint_grayscale_noise = background_noise[0:outpaint_height, :]

        rgb_outpaint_noisy = rgb_outpaint_noiseless + np.dstack(tuple([outpaint_grayscale_noise] * 3)).astype(np.uint8)

        return rgb_outpaint_noisy

    def _create_rgb_outpaint_noiseless_vert(
            self,  x: NDArray[int], y: NDArray[int], img_blur: NDArray[np.uint8],
            image_width: int, background_height: int, outpaint_height: int,
            location: Location) -> NDArray[np.uint8]:
        """
        Create the RGB outpaint without noise
        :param x (int): x coordinates of the data points
        :param y (int): y coordinates of the data points
        :param img_blur (np.ndarray): image with applied Gaussian blur
        :param image_width (int): width of the image
        :param background_height (int): height of the background area
        :param outpaint_height (int): height of the area to outpaint
        :param location (Location): location of the outpaint
        """
        # do not inject any noise in the outpaint area in this case
        channel_outpaints = list()
        for i in range(0, 3):
            data = self._get_interpolation_dataset(img_blur, background_height, image_width, i, location)

            # Create interpolator and use it to extrapolate the background data into the outpaint area
            channel_outpaint = self._extrapolate_color_channel(x, y, data, outpaint_height, image_width)
            channel_outpaints.append(channel_outpaint)

        rgb_outpaint_noiseless = np.dstack(tuple(channel_outpaints))
        return rgb_outpaint_noiseless

    def _extrapolate_rgb_background_with_gaussian_white_noise_vert(
            self, x: NDArray[int], y: NDArray[int],
            img: NDArray[np.uint8],
            image_width: int, background_height: int, outpaint_height: int, location: Location) -> NDArray[np.uint8]:
        """
        Extrapolate the RGB data to outpaint the specified area extending vertically
        :param x (int): x coordinates of the data points
        :param y (int): y coordinates of the data points
        :param img_blur (np.ndarray): image with applied Gaussian blur
        :param img (np.ndarray): image to outpaint
        :param image_width (int): height of the image
        :param background_height (int): width of the background area
        :param outpaint_height (int): width of the area to outpaint
        :param location (Location): location of the outpaint
        """
        if self.NOISE_CONSTRUCTION_METHOD == "color-channel-based":
            rgb_outpaint = self._add_white_noise_to_rgb_outpaint_via_color_channel_vert(
                x, y, img, image_width, background_height, outpaint_height, location)

        elif self.NOISE_CONSTRUCTION_METHOD == "luminance-based":
            rgb_outpaint = self._add_white_noise_to_rgb_outpaint_via_luminance_vert(
                x, y, image_width, background_height, outpaint_height, location)
        else:
            raise ValueError(f'Unknown noise construction method: {self.NOISE_CONSTRUCTION_METHOD}')

        return rgb_outpaint

    def _extrapolate_rgb_background_with_gaussian_colored_noise_vert(
            self, x: NDArray[int], y: NDArray[int],
            img: NDArray[np.uint8],
            image_width: int, background_height: int, outpaint_height: int, location: Location) -> NDArray[np.uint8]:
        """
        Extrapolate the RGB data to outpaint the specified area extending vertically and add gaussian colored noise
        :param x (int): x coordinates of the data points
        :param y (int): y coordinates of the data points
        :param img (np.ndarray): image to outpaint
        :param image_width (int): width of the image
        :param background_height (int): height of the background area
        :param outpaint_height (int): height of the area to outpaint
        :param location (Location): location of the outpaint
        """

        if self.NOISE_CONSTRUCTION_METHOD == "color-channel-based":
            rgb_outpaint = self._add_colored_noise_to_rgb_outpaint_via_color_channel_vert(
                x, y, img, image_width, background_height, outpaint_height, location)

        elif self.NOISE_CONSTRUCTION_METHOD == "luminance-based":
            rgb_outpaint = self._add_colored_noise_to_rgb_outpaint_via_luminance_vert(
                x, y, image_width, background_height, outpaint_height, location)
        else:
            raise ValueError(f'Unknown noise construction method: {self.NOISE_CONSTRUCTION_METHOD}')

        return rgb_outpaint

    def _extrapolate_rgb_background_vert(
            self, x: NDArray[int], y: NDArray[int],
            img: NDArray[np.uint8],
            image_width: int, background_height: int, outpaint_height: int, location: Location) -> NDArray[np.uint8]:
        """
        Extrapolate the RGB data to outpaint the specified area extending vertically
        :param x (int): x coordinates of the data points
        :param y (int): y coordinates of the data points
        :param img (np.ndarray): image to outpaint
        :param image_width (int): width of the image
        :param background_height (int): height of the background area
        :param outpaint_height (int): height of the area to outpaint
        :param location (Location): location of the outpaint
        """

        if self.NOISE_MODEL == "gradient-background":
            rgb_outpaint = self._create_rgb_outpaint_with_gradient_background_noise_vert(
                x, y, self._img_blur_noiseless_outpaint, image_width, background_height, outpaint_height, location)
        elif self.NOISE_MODEL == "gaussian-white":
            rgb_outpaint = self._extrapolate_rgb_background_with_gaussian_white_noise_vert(
                x, y, img, image_width, background_height, outpaint_height, location)
        elif self.NOISE_MODEL == "gaussian-colored":
            rgb_outpaint = self._extrapolate_rgb_background_with_gaussian_colored_noise_vert(
                x, y, img, image_width, background_height, outpaint_height, location)
        elif self.NOISE_MODEL == "none":
            rgb_outpaint = self._create_rgb_outpaint_noiseless_vert(
                x, y, self._img_blur_noiseless_outpaint, image_width, background_height, outpaint_height, location)
        else:
            raise ValueError(f'Unknown noise model: {self.NOISE_MODEL}')

        return rgb_outpaint

    @staticmethod
    def _get_interpolation_dataset(
            img: NDArray[np.uint8], background_height: int, background_width: int,
            color_chan_index: int, location: Location) -> NDArray[np.uint8]:
        """
        Get the dataset for the interpolation
        :param img (np.ndarray): image to outpaint
        :param background_height (int): height of the background area
        :param background_width (int): width of the background area
        :param color_chan_index (int): index of the color channel
        :param location (Location): location of the outpaint
        return interpolation dataset for single color channel (np.ndarray)
        """
        if location == Location.LEFT:
            return img[0:background_height, 0:background_width, color_chan_index]
        elif location == Location.RIGHT:
            return np.fliplr(img[0:background_height, -background_width:, color_chan_index])
        elif location == Location.TOP:
            return img[0:background_height, 0:background_width, color_chan_index]
        elif location == Location.BOTTOM:
            return np.flipud(img[-background_height:, 0:background_width, color_chan_index])

    def _inject_luminance_white_noise_horiz(self, rgb_outpaint_noiseless: NDArray[np.uint8],
                                            image_height: int, background_width: int,
                                            outpaint_width: int, location: Location) -> NDArray[np.uint8]:
        """
        Inject luminance-based gaussian white noise into the image
        :param img (np.ndarray): image to outpaint
        :param img_blur (np.ndarray): image with applied Gaussian blur
        :param background_height (int): height of the background area
        :param background_width (int): width of the background area
        :param outpaint_width (int): width of the area to outpaint
        :param location (Location): location of the outpaint
        return image with injected noise (np.ndarray)
        """

        background_noise = self._get_grayscale_noise_dataset(
            self._img_blur_gray_noise_estimate, self._img_gray_noise_estimate, image_height, background_width, location)

        outpaint_grayscale_noise = self._generate_channel_white_noise_horiz(
            background_noise, image_height, background_width, outpaint_width)

        if self.NOISE_INJECTION_METHOD == "noise-factor-estimate":
            # an alternative way to estimate the luminescence noise using the grayscale noiseless outpaint
            # and noise factor; less biased than the additive method

            gray_outpaint_noiseless = cv2.cvtColor(rgb_outpaint_noiseless.astype(np.uint8), cv2.COLOR_BGR2GRAY)

            noise_factor = np.divide(outpaint_grayscale_noise,  gray_outpaint_noiseless, where=gray_outpaint_noiseless != 0)

            rgb_outpaint_noisy = np.multiply(rgb_outpaint_noiseless, np.ones(shape=rgb_outpaint_noiseless.shape) +
                                         np.dstack(tuple([noise_factor] * 3))).astype(np.uint8)

        elif self.NOISE_INJECTION_METHOD == "additive":
            # simply add the grayscale noise to each color channel
            # drawback - may introduce bias

            rgb_outpaint_noisy = rgb_outpaint_noiseless + np.dstack(tuple([outpaint_grayscale_noise] * 3)).astype(np.uint8)

        else:
            raise ValueError(f'Unknown noise injection method: {self.NOISE_INJECTION_METHOD}')

        return rgb_outpaint_noisy

    def _get_outpaint_gaussian_colored_noise_grayscale_horiz(self, image_height: int, background_width: int,
                                                         outpaint_width: int, location: Location) -> NDArray[np.uint8]:
        """
        Get the outpaint region with colored Gaussian noise
        :param image_height (int): height of the image
        :param background_width (int): width of the background area
        :param outpaint_width (int): width of the area to outpaint
        :param location (Location): location of the outpaint
        return outpaint region with colored noise (np.ndarray)
        """
        def get_outpaint_slice():
            """
            a helper function to generate a slice of the outpaint region with colored noise
            Note: a slice is a sequences of blocks containing noise with the size of the colored noise block
            """
            # number of blocks which can be stacked horizontally in the outpaint region
            number_of_blocks_horiz_outpaint = outpaint_width // self.COLORED_NOISE_BLOCK_SIZE
            # leftover pixels less than one block size of width in the outpaint region
            remainder_horiz_outpaint = outpaint_width % self.COLORED_NOISE_BLOCK_SIZE

            grayscale_white_noise = (
                np.random.normal(0, 1, (row_count, number_of_blocks_horiz_outpaint)).astype(int))
            horiz_slice = (
                np.real(np.dot(cov_sqrt_gray, grayscale_white_noise)).reshape(self.COLORED_NOISE_BLOCK_SIZE, -1))

            if remainder_horiz_outpaint > 0:
                grayscale_white_noise = np.random.normal(0, 1, (row_count, 1)).astype(int)
                outpaint_grayscale_noise_block = np.real(np.dot(cov_sqrt_gray, grayscale_white_noise)).reshape(
                    self.COLORED_NOISE_BLOCK_SIZE, -1)
                horiz_slice = np.hstack((horiz_slice, outpaint_grayscale_noise_block[:, 0:remainder_horiz_outpaint]))
            return horiz_slice

        background_noise = self._get_grayscale_noise_dataset(
            self._img_blur_gray_noise_estimate, self._img_gray_noise_estimate, image_height, background_width, location)

        # number of blocks which can be stacked horizontally in the gradient background region
        number_of_blocks_horiz = background_width // self.COLORED_NOISE_BLOCK_SIZE

        # number of blocks which can be stacked vertically in the outpaint region
        number_of_blocks_vert = image_height // self.COLORED_NOISE_BLOCK_SIZE
        # leftover pixels less than one block size of height in the gradient background region
        remainder_vert = image_height % self.COLORED_NOISE_BLOCK_SIZE

        row_count = self.COLORED_NOISE_BLOCK_SIZE * self.COLORED_NOISE_BLOCK_SIZE

        # gather all available blocks in all slices in the gradient background region noise as a training dataset
        # Note: a slice is a sequences of blocks containing noise with the size of self.COLORED_NOISE_BLOCK_SIZE
        slice_array = None
        for i in range(0, number_of_blocks_vert):
            background_noise_slice = background_noise[
                                     i * self.COLORED_NOISE_BLOCK_SIZE: (i+1) * self.COLORED_NOISE_BLOCK_SIZE,
                                     :number_of_blocks_horiz * self.COLORED_NOISE_BLOCK_SIZE]
            if slice_array is not None:
                slice_array = np.hstack((slice_array, background_noise_slice.T.reshape(1, -1).reshape(row_count, -1)))
            else:
                slice_array = background_noise_slice.T.reshape(1, -1).reshape(row_count, -1)

        # introducing colored Gaussian noise as the dot product of the square root of the covariance matrix of the
        # noise in the training dataset and a white Gaussian noise with empirically computed mean and std dev.
        cov_matrix_gray_noise = np.cov(slice_array)
        cov_sqrt_gray = self.STANDARD_DEVIATION_FACTOR * linalg.sqrtm(cov_matrix_gray_noise)

        # construct all slices with colored noise in the outpaint region
        outpaint_grayscale_noise = None
        for i in range(0, number_of_blocks_vert):
            outpaint_slice = get_outpaint_slice()
            if outpaint_grayscale_noise is not None:
                outpaint_grayscale_noise = np.vstack((outpaint_grayscale_noise, outpaint_slice))
            else:
                outpaint_grayscale_noise = outpaint_slice
        if remainder_vert > 0:
            outpaint_slice = get_outpaint_slice()
            outpaint_grayscale_noise = np.vstack((outpaint_grayscale_noise, outpaint_slice[0:remainder_vert, :]))

        return outpaint_grayscale_noise

    def _inject_luminance_colored_noise_horiz(self, rgb_outpaint_noiseless: NDArray[np.uint8],
                                              image_height: int, background_width: int,
                                              outpaint_width: int, location: Location) -> NDArray[np.uint8]:
        """
        Inject luminance-based colored gaussian noise into the image
        :param background_height (int): height of the background area
        :param background_width (int): width of the background area
        :param outpaint_width (int): width of the area to outpaint
        :param location (Location): location of the outpaint
        return image with injected noise (np.ndarray)
        """
        outpaint_grayscale_noise = self._get_outpaint_gaussian_colored_noise_grayscale_horiz(
             image_height, background_width, outpaint_width, location)

        if self.NOISE_INJECTION_METHOD == "noise-factor-estimate":
            # an alternative way to estimate the luminescence noise using the grayscale noiseless outpaint
            # and noise factor; less biased than the additive method

            gray_outpaint_noiseless = cv2.cvtColor(rgb_outpaint_noiseless.astype(np.uint8), cv2.COLOR_BGR2GRAY)

            noise_factor = np.divide(outpaint_grayscale_noise, gray_outpaint_noiseless, where=gray_outpaint_noiseless != 0)

            rgb_outpaint_noisy = np.multiply(rgb_outpaint_noiseless, np.ones(shape=rgb_outpaint_noiseless.shape) +
                                         np.dstack(tuple([noise_factor] * 3))).astype(np.uint8)

        elif self.NOISE_INJECTION_METHOD == "additive":
            # simply add the grayscale noise to each color channel
            # drawback - may introduce bias

            rgb_outpaint_noisy = rgb_outpaint_noiseless + np.dstack(tuple([outpaint_grayscale_noise] * 3))

        else:
            raise ValueError(f'Unknown noise injection method: {self.NOISE_INJECTION_METHOD}')

        if self.APPLY_BLUR_ON_PAINT:
            return cv2.GaussianBlur(rgb_outpaint_noisy, self.COLORED_NOISE_BLUR_KERNEL_SIZE, self.COLORED_NOISE_BLUR_SIGMA)
        else:
            return rgb_outpaint_noisy

    def _inject_luminance_white_noise_vert(self, rgb_outpaint_noiseless: NDArray[np.uint8],
                                           image_width: int, background_height: int,
                                           outpaint_height: int, location: Location) -> NDArray[np.uint8]:
        """
        Inject luminance-based gaussian white noise into the image
        :param background_width (int): width of the background area
        :param background_height (int): height of the background area
        :param outpaint_height (int): height of the area to outpaint
        :param location (Location): location of the outpaint
        return image with injected noise (np.ndarray)
        """
        background_noise = self._get_grayscale_noise_dataset(
            self._img_blur_gray_noise_estimate, self._img_gray_noise_estimate, background_height, image_width, location)

        outpaint_grayscale_noise = self._generate_channel_white_noise_vert(background_noise, image_width,
                                                                       background_height, outpaint_height)

        if self.NOISE_INJECTION_METHOD == "noise-factor-estimate":
            # an alternative way to estimate the luminescence noise using the grayscale noiseless outpaint
            # and noise factor; less biased than the additive method

            gray_outpaint_noiseless = cv2.cvtColor(rgb_outpaint_noiseless.astype(np.uint8), cv2.COLOR_BGR2GRAY)

            noise_factor = np.divide(outpaint_grayscale_noise,  gray_outpaint_noiseless, where=gray_outpaint_noiseless != 0)

            rgb_outpaint_noisy = np.multiply(rgb_outpaint_noiseless, np.ones(shape=rgb_outpaint_noiseless.shape) +
                                         np.dstack(tuple([noise_factor] * 3))).astype(np.uint8)

        elif self.NOISE_INJECTION_METHOD == "additive":
            # simply add the grayscale noise to each color channel
            # drawback - may introduce bias

            rgb_outpaint_noisy = rgb_outpaint_noiseless + np.dstack(tuple([outpaint_grayscale_noise] * 3)).astype(np.uint8)

        else:
            raise ValueError(f'Unknown noise injection method: {self.NOISE_INJECTION_METHOD}')

        return rgb_outpaint_noisy

    def _get_outpaint_gaussian_colored_noise_grayscale_vert(self, image_width: int, background_height: int,
                                                        outpaint_height: int, location: Location) -> NDArray[np.uint8]:
        """
        Get the outpaint region with colored Gaussian noise
        :param image_width (int): width of the image
        :param background_height (int): height of the background area
        :param outpaint_height (int): height of the area to outpaint
        :param location (Location): location of the outpaint
        return outpaint region with colored noise (np.ndarray)
        """
        def get_outpaint_slice():
            """
            a helper function to generate a slice of the outpaint region with colored noise
            Note: a slice is a sequences of blocks containing noise with the size of the colored noise block
            """
            # number of blocks which can be stacked horizontally in the outpaint region
            number_of_blocks_vert_outpaint = outpaint_height // self.COLORED_NOISE_BLOCK_SIZE
            # leftover pixels less than one block size of width in the outpaint region
            remainder_vert_outpaint = outpaint_height % self.COLORED_NOISE_BLOCK_SIZE
            grayscale_white_noise = (
                np.random.normal(0, 1, (number_of_blocks_vert_outpaint, col_count)).astype(int))
            vert_slice = (
                np.real(np.dot(grayscale_white_noise, cov_sqrt_gray)).reshape(-1, self.COLORED_NOISE_BLOCK_SIZE))

            if remainder_vert_outpaint > 0:
                grayscale_white_noise = np.random.normal(0, 1, (1, col_count)).astype(int)

                outpaint_grayscale_noise_block = np.real(np.dot(grayscale_white_noise, cov_sqrt_gray)).reshape(
                    -1, self.COLORED_NOISE_BLOCK_SIZE)
                vert_slice = np.vstack((vert_slice, outpaint_grayscale_noise_block[0:remainder_vert_outpaint, :]))
            return vert_slice

        background_noise = self._get_grayscale_noise_dataset(
             self._img_blur_gray_noise_estimate, self._img_gray_noise_estimate, background_height, image_width, location)

        # number of blocks which can be stacked vertically in the gradient background region
        number_of_blocks_vert = background_height // self.COLORED_NOISE_BLOCK_SIZE
        # leftover pixels less than one block size of height in the gradient background region
        remainder_vert = background_height % self.COLORED_NOISE_BLOCK_SIZE

        # number of blocks which can be stacked horizontally in the outpaint region
        number_of_blocks_horiz = image_width // self.COLORED_NOISE_BLOCK_SIZE
        # leftover pixels less than one block size of width in the gradient background region
        remainder_horiz = image_width % self.COLORED_NOISE_BLOCK_SIZE

        col_count = self.COLORED_NOISE_BLOCK_SIZE * self.COLORED_NOISE_BLOCK_SIZE

        if number_of_blocks_vert == 0:
            # edge case in which the block size is larger than the background height
            # in this case we tile the gradient background vertically until it reaches the height of the block
            stack_count = self.COLORED_NOISE_BLOCK_SIZE // remainder_vert
            stack_remainder = self.COLORED_NOISE_BLOCK_SIZE % remainder_vert
            background_noise = np.tile(background_noise, (stack_count, 1))
            background_noise = np.vstack((background_noise, background_noise[0:stack_remainder, :]))
            number_of_blocks_vert = 1
            remainder_vert = 0

        # gather all available blocks in all slices in the gradient background region as a training dataset
        # Note: a slice is a sequences of blocks containing noise with the size of self.COLORED_NOISE_BLOCK_SIZE
        slice_array = None
        for i in range(0, number_of_blocks_horiz):
            background_noise_slice = background_noise[
                                     :number_of_blocks_vert * self.COLORED_NOISE_BLOCK_SIZE,
                                     i * self.COLORED_NOISE_BLOCK_SIZE: (i+1) * self.COLORED_NOISE_BLOCK_SIZE]

            if slice_array is not None:
                slice_array = np.vstack((slice_array, background_noise_slice.reshape(-1, col_count)))
            else:
                slice_array = background_noise_slice.reshape(-1, col_count)

        # introducing colored Gaussian noise as the dot product of the square root of the covariance matrix of the
        # noise in the training dataset and a white Gaussian noise with empirically computed mean and std dev.
        cov_matrix_gray_noise = np.cov(slice_array.T)
        cov_sqrt_gray = self.STANDARD_DEVIATION_FACTOR * linalg.sqrtm(cov_matrix_gray_noise)

        # construct all slices with colored noise in the outpaint region
        outpaint_grayscale_noise = None
        for i in range(0, number_of_blocks_horiz):
            outpaint_slice = get_outpaint_slice()
            if outpaint_grayscale_noise is not None:
                outpaint_grayscale_noise = np.hstack((outpaint_grayscale_noise, outpaint_slice))
            else:
                outpaint_grayscale_noise = outpaint_slice
        if remainder_horiz > 0:
            outpaint_slice = get_outpaint_slice()
            outpaint_grayscale_noise = np.hstack((outpaint_grayscale_noise, outpaint_slice[:, 0:remainder_horiz]))

        return outpaint_grayscale_noise

    def _inject_luminance_colored_noise_vert(self, rgb_outpaint_noiseless: NDArray[np.uint8],
                                             image_width: int, background_height: int,
                                             outpaint_height: int, location: Location) -> NDArray[np.uint8]:
        """
        Inject luminance-based noise into the image
        :param background_width (int): width of the background area
        :param background_height (int): height of the background area
        :param outpaint_height (int): height of the area to outpaint
        :param location (Location): location of the outpaint
        return image with injected noise (np.ndarray)
        """

        outpaint_grayscale_noise = self._get_outpaint_gaussian_colored_noise_grayscale_vert(
            image_width, background_height, outpaint_height, location)

        if self.NOISE_INJECTION_METHOD == "noise-factor-estimate":
            # an alternative way to estimate the luminescence noise using the grayscale noiseless outpaint
            # and noise factor; less biased than the additive method

            gray_outpaint_noiseless = cv2.cvtColor(rgb_outpaint_noiseless.astype(np.uint8), cv2.COLOR_BGR2GRAY)

            noise_factor = np.divide(outpaint_grayscale_noise, gray_outpaint_noiseless, where=gray_outpaint_noiseless != 0)

            rgb_outpaint_noisy = np.multiply(rgb_outpaint_noiseless, np.ones(shape=rgb_outpaint_noiseless.shape) +
                                         np.dstack(tuple([noise_factor] * 3)))

        elif self.NOISE_INJECTION_METHOD == "additive":
            # simply add the grayscale noise to each color channel
            # drawback - may introduce bias

            rgb_outpaint_noisy = rgb_outpaint_noiseless + np.dstack(tuple([outpaint_grayscale_noise] * 3))

        else:
            raise ValueError(f'Unknown noise injection method: {self.NOISE_INJECTION_METHOD}')

        if self.APPLY_BLUR_ON_PAINT:
            return cv2.GaussianBlur(rgb_outpaint_noisy, self.COLORED_NOISE_BLUR_KERNEL_SIZE, self.COLORED_NOISE_BLUR_SIGMA)
        else:
            return rgb_outpaint_noisy

    @staticmethod
    def _get_noise_dataset(img_blur: NDArray[np.uint8], img: NDArray[np.uint8],
                           background_height: int, background_width: int,
                           color_chan_index: int, location: Location) -> NDArray[Any] | Any | None:
        """
        Get the dataset for the interpolation
        :param img (np.ndarray): image to outpaint
        :param background_height (int): height of the background area
        :param background_width (int): width of the background area
        :param color_chan_index (int): index of the color channel
        :param location (Location): location of the outpaint
        return interpolation dataset (np.ndarray)
        """
        if location == Location.LEFT:
            return np.subtract(img[0:background_height, 0:background_width, color_chan_index].astype(int),
                               img_blur[0:background_height, 0:background_width, color_chan_index].astype(int))
        elif location == Location.RIGHT:
            return np.fliplr(np.subtract(img[0:background_height, -background_width:, color_chan_index].astype(int),
                                         img_blur[0:background_height, -background_width:, color_chan_index].astype(int)
                                         ))
        elif location == Location.TOP:
            return np.subtract(img[0:background_height, 0:background_width, color_chan_index].astype(int),
                               img_blur[0:background_height, 0:background_width, color_chan_index].astype(int))
        elif location == Location.BOTTOM:
            return np.flipud(np.subtract(img[-background_height:, 0:background_width, color_chan_index].astype(int),
                                         img_blur[-background_height:, 0:background_width, color_chan_index].astype(int)
                                         ))

    @staticmethod
    def _get_grayscale_noise_dataset(img_blur_gray: NDArray[np.uint8], img_gray: NDArray[np.uint8],
                                     background_height: int, background_width: int,
                                     location: Location) -> NDArray[Any] | Any | None:
        """
        Get the dataset for the interpolation
        :param img_blur_gray (np.ndarray): grayscale image with applied Gaussian blur
        :param img_gray (np.ndarray): grayscale image to outpaint
        :param background_height (int): height of the background area
        :param background_width (int): width of the background area
        :param color_chan_index (int): index of the color channel
        :param location (Location): location of the outpaint
        return interpolation dataset (np.ndarray)
        """
        if location == Location.LEFT:
            return np.subtract(img_gray[0:background_height, 0:background_width].astype(int),
                               img_blur_gray[0:background_height, 0:background_width].astype(int))
        elif location == Location.RIGHT:
            return np.fliplr(np.subtract(img_gray[0:background_height, -background_width:].astype(int),
                                         img_blur_gray[0:background_height, -background_width:].astype(int)
                                         ))
        elif location == Location.TOP:
            return np.subtract(img_gray[0:background_height, 0:background_width].astype(int),
                               img_blur_gray[0:background_height, 0:background_width].astype(int))
        elif location == Location.BOTTOM:
            return np.flipud(np.subtract(img_gray[-background_height:, 0:background_width].astype(int),
                                         img_blur_gray[-background_height:, 0:background_width].astype(int)
                                         ))

    def _generate_channel_white_noise_horiz(self, noise_data, image_height, background_width, outpaint_width)\
            -> NDArray[np.uint8]:
        """
        Generate the noise data to outpaint the specified area horizontally
        :param background_noise: noise data
        :param image_height: height of the image
        :param background_width: width of the background area
        :param outpaint_width: width of the area to outpaint
        :return ndarray: extrapolated noise data in the outpaint region
        """
        mean, std = self._estimate_gaussian_white_noise_distribution_horiz(noise_data, background_width)
        return np.random.normal(mean, std, (image_height, outpaint_width)).astype(int)

    def _generate_channel_white_noise_vert(self, noise_data, image_width, background_height, outpaint_height)\
            -> NDArray[np.uint8]:
        """
        Generate the noise data to outpaint the specified area vertically
        :param background_noise: noise data
        :param image_width: width of the image
        :param background_height: height of the background area
        :param outpaint_height: height of the area to outpaint
        :return ndarray: extrapolated noise data in the outpaint region
        """
        mean, std = self._estimate_gaussian_white_noise_distribution_vert(noise_data, background_height)
        return np.random.normal(mean, std, (outpaint_height, image_width)).astype(int)

    def _estimate_gaussian_white_noise_distribution_horiz(self, noise_data: NDArray[np.uint8], region_width: int) \
            -> tuple[np.floating, np.floating]:
        """
        Estimate the gaussian white noise distribution for the specified area horizontally
        :param noise_data: noise data
        :param outpaint_width: width of the area to outpaint
        :return: the estimated parameters (mean and standard dev) of the Gaussian noise distribution
        """
        column_period = region_width // self.SAMPLE_COLUMNS_COUNT

        means = list()
        stds = list()
        for i in range(self.SAMPLE_COLUMNS_COUNT):
            column = noise_data[:, column_period // 2 + i * column_period]
            means.append(np.mean(column))
            stds.append(np.std(column))
        return np.median(means), np.median(stds)

    def _estimate_gaussian_white_noise_distribution_vert(self, noise_data: NDArray[np.uint8],
                                                         region_height: int) -> tuple[np.floating, np.floating]:
        """
        Estimate the gaussian white noise distribution for the specified area vertically
        :param noise_data: noise data
        :param outpaint_height: height of the area to outpaint
        :return: the estimated parameters (mean and standard dev) of the Gaussian noise distribution
        """
        row_period = region_height // self.SAMPLE_ROWS_COUNT
        means = list()
        stds = list()
        for i in range(self.SAMPLE_ROWS_COUNT):
            row = noise_data[row_period // 2 + i * row_period, :]
            means.append(np.mean(row))
            stds.append(np.std(row))

        return np.median(means), np.median(stds)

    def _extrapolate_color_channel(self, x:  NDArray[int], y:  NDArray[int], data: NDArray[np.uint8],
                                   outpaint_height: int, outpaint_width: int) -> NDArray[np.uint8]:
        """
        Extrapolate the color channel data to outpaint the specified area
        :param x: x coordinates of the data points
        :param y: y coordinates of the data points
        :param data: color channel data
        :param outpaint_height: height of the area to outpaint
        :param outpaint_width: width of the area to outpaint
        """
        if self.INTERPOLATION_METHOD != "stacked":
            # Extrapolate
            points_to_extrapolate = np.zeros(
                shape=(outpaint_height * outpaint_width, 2), dtype=int
            )

            # Create interpolator
            interp = RegularGridInterpolator(
                (x, y), data, bounds_error=False, fill_value=None, method=self.INTERPOLATION_METHOD
            )

            for i in range(0, outpaint_width):
                for j in range(0, outpaint_height):
                    points_to_extrapolate[i * outpaint_height + j, :] = [j, i]

            extrapolated_values = interp(points_to_extrapolate)

            return extrapolated_values.reshape(outpaint_height, outpaint_width, order='F')
        else:
            return self._extrapolate_color_channel_via_stacking(x, y, data, outpaint_height, outpaint_width)

    def _extrapolate_color_channel_via_stacking(self, x:  NDArray[int], y:  NDArray[int], data: NDArray[np.uint8],
                                                outpaint_height: int, outpaint_width: int) -> NDArray[np.uint8]:
        """
        Extrapolate the color channel data to outpaint the specified area using the stacking method
        :param x: x coordinates of the data points
        :param y: y coordinates of the data points
        :param data: color channel data
        :param outpaint_height: height of the area to outpaint
        :param outpaint_width: width of the area to outpaint
        """
        if self.COLUMN_SELECTION_METHOD == "nearest":
            return self._extrapolate_color_channel_via_stacking_nearest(
                x, y, data, outpaint_height, outpaint_width
            )
        elif self.COLUMN_SELECTION_METHOD == "mean":
            return self._extrapolate_color_channel_via_stacking_mean(
                x, y, data, outpaint_height, outpaint_width
            )
        elif self.COLUMN_SELECTION_METHOD == "median":
            return self._extrapolate_color_channel_via_stacking_median(
                x, y, data, outpaint_height, outpaint_width
            )
        else:
            raise ValueError(f'Unknown column selection method: {self.COLUMN_SELECTION_METHOD}')

    @staticmethod
    def _extrapolate_color_channel_via_stacking_nearest(
            x:  NDArray[int], y:  NDArray[int], data: NDArray[np.uint8],
            outpaint_height: int, outpaint_width: int) -> NDArray[np.uint8]:
        """
        Extrapolate the color channel data to outpaint the specified area using the stacking method
        :param x: x coordinates of the data points
        :param y: y coordinates of the data points
        :param data: color channel data
        :param outpaint_height: height of the area to outpaint
        :param outpaint_width: width of the area to outpaint
        """
        if y[0] > 0:
            # we have horizontal outpaint
            # stack the first column of the data horizontally
            return np.tile(data[:, 0], (outpaint_width, 1)).T
        else:
            # we have vertical outpaint
            # stack the first row of the data vertically
            return np.tile(data[0, :], (outpaint_height, 1))

    def _extrapolate_color_channel_via_stacking_mean(
            self, x:  NDArray[int], y:  NDArray[int], data: NDArray[np.uint8],
            outpaint_height: int, outpaint_width: int) -> NDArray[np.uint8]:
        """
        Extrapolate the color channel data to outpaint the specified area using the stacking method
        :param x: x coordinates of the data points
        :param y: y coordinates of the data points
        :param data: color channel data
        :param outpaint_height: height of the area to outpaint
        :param outpaint_width: width of the area to outpaint
        """
        if y[0] > 0:
            cols = data[:, 0:int(self.NUMER_OF_COLUMNS_TO_CONSIDER)]
            mean_col = np.mean(cols, axis=1)
            # we have horizontal outpaint
            # stack the first column of the data horizontally
            return np.tile(mean_col, (outpaint_width, 1)).T
        else:
            rows = data[0:int(self.NUMER_OF_ROWS_TO_CONSIDER), :]
            mean_row = np.mean(rows, axis=0)
            # we have vertical outpaint
            # stack the first row of the data vertically
            return np.tile(mean_row, (outpaint_height, 1))

    def _extrapolate_color_channel_via_stacking_median(
            self, x:  NDArray[int], y:  NDArray[int], data: NDArray[np.uint8],
            outpaint_height: int, outpaint_width: int) -> NDArray[np.uint8]:
        """
        Extrapolate the color channel data to outpaint the specified area using the stacking method
        :param x: x coordinates of the data points
        :param y: y coordinates of the data points
        :param data: color channel data
        :param outpaint_height: height of the area to outpaint
        :param outpaint_width: width of the area to outpaint
        """
        if y[0] > 0:
            cols = data[:, 0:int(self.NUMER_OF_COLUMNS_TO_CONSIDER)]
            median_col = np.median(cols, axis=1)
            # we have horizontal outpaint
            # stack the first column of the data horizontally
            return np.tile(median_col, (outpaint_width, 1)).T
        else:
            rows = data[0:int(self.NUMER_OF_ROWS_TO_CONSIDER), :]
            median_row = np.median(rows, axis=0)
            # we have vertical outpaint
            # stack the first row of the data vertically
            return np.tile(median_row, (outpaint_height, 1))
