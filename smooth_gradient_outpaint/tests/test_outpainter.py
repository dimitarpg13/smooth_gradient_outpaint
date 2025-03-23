import os
import unittest
import warnings
from collections import namedtuple
from smooth_gradient_outpaint.outpainter import (OutpainterConfig, Outpainter, ShadowDetector, ShadowRemover, Location,
                                                 find_largest_smooth_background_dimensions,
                                                 compute_outpaint_area_dimensions)
import numpy as np
from PIL import Image
import cv2


config = OutpainterConfig()

SAMPLE_DATASET_PATH = "tests/images"

NUMBER_OF_SAMPLE_IMAGES = 1

# Outpainter Test Parameters
#
#   image_name (str): the file name of the current image under test
#
#   gradient_background_width_left (int): the width of the gradient background on the left of the object of interest
#
#   gradient_background_width_right (int): the width of the gradient background on the right of the object of interest
#
#   gradient_background_height_top (int): the height of the gradient background on the top of the object of interest
#
#   gradient_background_height_bottom (int): the height of the gradient background on the bottom of
#   the object of interest
#
#   outpaint_left_width (int): the width of the outpainted region to add to the left side of the image
#
#   outpaint_right_width (int): the width of the outpainted region to add to the right side of the image
#
#   outpaint_top_height (int): the height of the outpainted region to add to the top of the image
#
#   outpaint_bottom_height (int): the height of the outpainted to add to the bottom of the image


data_param_list_outpainter = [
    # (
    #     "<image_name_name>",
    #     109,  # gradient_background_width_left
    #     0,  # gradient_background_width_right
    #     0,  # gradient_background_height_top
    #     0,  # gradient_background_height_bottom
    #     200,   # outpaint_left_width
    #     0,  # outpaint_right_width
    #     0,  # outpaint_top_height
    #     0,  # outpaint_bottom_height
    # ),
]


class OutpainterTestCase(unittest.TestCase):

    def test_outpainter(self):
        for (
            image_name,
            gradient_background_width_left,
            gradient_background_width_right,
            gradient_background_height_top,
            gradient_background_height_bottom,
            outpaint_left_width,
            outpaint_right_width,
            outpaint_top_height,
            outpaint_bottom_height,
        ) in data_param_list_outpainter:
            with self.subTest(
                image_name=image_name,
                gradient_background_width_left=gradient_background_width_left,
                gradient_background_width_right=gradient_background_width_right,
                gradient_background_height_top=gradient_background_height_top,
                gradient_background_height_bottom=gradient_background_height_bottom,
                outpaint_left_width=outpaint_left_width,
                outpaint_right_width=outpaint_right_width,
                outpaint_top_height=outpaint_top_height,
                outpaint_bottom_height=outpaint_bottom_height,
            ):
                print(
                    f"\nProcessing image {image_name}...\n"
                )
                full_path = os.path.join(
                    config.SMOOTH_GRADIENT_OUTPAINTER_HOME, SAMPLE_DATASET_PATH, image_name + ".PNG"
                )

                PIL_img = Image.open(str(full_path))
                profile = PIL_img.info.get("icc_profile")
                img = np.array(PIL_img)
                img_height, img_width, c = img.shape
                # remove the alpha channel if such is present
                if c == 4:
                    img = img[:, :, :3]

                background_bbox = {
                    Location.LEFT: gradient_background_width_left,
                    Location.RIGHT: gradient_background_width_right,
                    Location.TOP: gradient_background_height_top,
                    Location.BOTTOM: gradient_background_height_bottom,
                }

                outpaint_bbox = {
                    Location.LEFT: outpaint_left_width,
                    Location.RIGHT: outpaint_right_width,
                    Location.TOP: outpaint_top_height,
                    Location.BOTTOM: outpaint_bottom_height,
                }

                outpainter = Outpainter()

                outpainted_img = outpainter.paint(img, background_bbox, outpaint_bbox)
                full_path_with_outpaint = os.path.join(
                    config.SMOOTH_GRADIENT_OUTPAINTER_HOME, SAMPLE_DATASET_PATH,
                    image_name + "_" + str(outpaint_left_width) + "_" + str(outpaint_right_width) +
                    "_" + str(outpaint_top_height) + "_" + str(outpaint_bottom_height) + "_outpainted" + ".PNG"
                )

                PIL_outpainted_img = Image.fromarray(outpainted_img)
                if profile:
                    # set the icc profile if it exists in order to avoid fading of the colors
                    PIL_outpainted_img.info['icc_profile'] = profile
                PIL_outpainted_img.save(str(full_path_with_outpaint))

                background_dimensions = find_largest_smooth_background_dimensions(outpainted_img)

                delta = 10
                if abs(gradient_background_width_left + outpaint_left_width - background_dimensions[Location.LEFT]) > delta:
                    warnings.warn(
                        f"Image {image_name}: background width on the left differs too much "
                        f"from the position OOI x coordinate! "
                        f"Check for shadow interference!\n"
                        f"gradient_background_width_left + fill_left_width: "
                        f"{gradient_background_width_left + outpaint_left_width}\n"
                        f"background_dimensions[Location.LEFT]: {background_dimensions[Location.LEFT]}"
                    )

                if abs(gradient_background_width_right + fill_right_width - background_dimensions[Location.RIGHT]) > delta:
                    warnings.warn(
                        f"Image {image_name}: background width from the right differs too much "
                        f"from the position of the OOI x coordinate + OOI width! "
                        f"Check for shadow interference!\n"
                        f"gradient_background_width_right + fill_right_width: "
                        f"{gradient_background_width_right + outpaint_right_width}\n"
                        f"background_dimensions[Location.RIGHT]: {background_dimensions[Location.RIGHT]}"
                    )

                if abs(gradient_background_height_top + outpaint_top_height - background_dimensions[Location.TOP]) > delta:
                    warnings.warn(
                        f"Image {image_name}: background height on the top differs too much "
                        f"from the position of the OOI y coordinate! "
                        f"Check for shadow interference!\n"
                        f"{gradient_background_height_top + outpaint_top_height}\n"
                        f"background_dimensions[Location.TOP]: {background_dimensions[Location.TOP]}"
                    )

                if abs(gradient_background_height_bottom + outpaint_bottom_height -
                       background_dimensions[Location.BOTTOM]) > delta:
                    warnings.warn(
                        f"Image {image_name}: background height on the bottom differs too much "
                        f"from the position of the OOI y coordinate + OOI height! "
                        f"Check for shadow interference!\n"
                        f"{gradient_background_height_bottom + outpaint_bottom_height}\n"
                        f"background_dimensions[Location.BOTTOM]: {background_dimensions[Location.BOTTOM]}"
                    )


# Outpaint Dimensions Parameters
#
#   image_name (str): the file name of the current image under test
#
#   original_aspect_ratio_a (int): the numerator in the aspect ratio of the image
#
#   original_aspect_ratio_b (int): the denominator in the aspect ratio of the image
#
#   target_aspect_ratio_a (int): the numerator of the desired aspect ratio which the fill should achieve
#
#   target_aspect_ratio_b (int): the denominator of the desired aspect ratio which the fill should achieve
#
#   object_of_interest_x_coord (int): the x coordinate of the upper left corner of the object of interest
#
#   object_of_interest_y_coord (int): the y coordinate of the upper left corner of the object of interest
#
#   object_of_interest_width (int): the width of the object of interest
#
#   object_of_interest_height (int): the height of the object of interest
#
#   outpaint_left (int): the amount of padding to add to the left side of the image
#
#   outpaint_right (int): the amount of padding to add to the right side of the image
#
#   outpaint_top (int): the amount of padding to add to the top of the image
#
#   outpaint_bottom (int): the amount of padding to add to the bottom of the image
#
#   outpaint_horizontal_offset_object_of_interest (int): the offset of the middle of the object of interest
#   in horizontal direction from the middle of the image
#
#   outpaint_offset_vertical_object_of_interest (int):  the offset of the middle of object of interest in vertical direction
#   from the middle of the image

#   outpaint_offset_horizontal_center_of_mass (int): the offset of the center of mass of the object of interest with respect
#   to the middle of the image in horizontal direction - NOT USED //TODO: implement it
#
#   outpaint_offset_vertical_center_of_mass (int): the offset of the center of mass of the object of interest with respect
#   to the middle of the image in vertical direction - NOT USED //TODO: implement it
#

data_param_list_outpaint_dimensions = [
    # (
    #     "<image_file_name>",
    #     4,  # original_aspect_ratio_a
    #     5,  # original_aspect_ratio_b
    #     1,  # target_aspect_ratio_a
    #     1,  # target_aspect_ratio_b
    #     25,  # object_of_interest_x_coord
    #     145,  # object_of_interest_y_coord
    #     725,  # object_of_interest_width
    #     800,  # object_of_interest_height
    #     0,  # outpaint_left
    #     0,  # outpaint_right
    #     0,  # outpaint_top
    #     0,  # outpaint_bottom
    #     0,  # outpaint_horizontal_offset_object_of_interest
    #     0,  # outpaint_offset_vertical_object_of_interest
    # ),
]


class OutpaintDimensionsTestCase(unittest.TestCase):

    def test_compute_fill_dimensions(self):
        for (
            image_name,
            original_aspect_ratio_a,
            original_aspect_ratio_b,
            target_aspect_ratio_a,
            target_aspect_ratio_b,
            object_of_interest_x_coord,
            object_of_interest_y_coord,
            object_of_interest_width,
            object_of_interest_height,
            expected_outpaint_left,
            expected_outpaint_right,
            expected_outpaint_top,
            expected_outpaint_bottom,
            outpaint_horizontal_offset_object_of_interest,
            outpaint_offset_vertical_object_of_interest,
        ) in data_param_list_outpaint_dimensions:
            with self.subTest(
                image_name=image_name,
                original_aspect_ratio_a=original_aspect_ratio_a,
                original_aspect_ratio_b=original_aspect_ratio_b,
                target_aspect_ratio_a=target_aspect_ratio_a,
                target_aspect_ratio_b=target_aspect_ratio_b,
                object_of_interest_x_coord=object_of_interest_x_coord,
                object_of_interest_y_coord=object_of_interest_y_coord,
                object_of_interest_width=object_of_interest_width,
                object_of_interest_height=object_of_interest_height,
                expected_outpaint_left=expected_outpaint_left,
                expected_outpaint_right=expected_outpaint_right,
                expected_outpaint_top=expected_outpaint_top,
                expected_outpaint_bottom=expected_outpaint_bottom,
                outpaint_horizontal_offset_object_of_interest=outpaint_horizontal_offset_object_of_interest,
                outpaint_offset_vertical_object_of_interest=outpaint_offset_vertical_object_of_interest,
            ):
                print(
                    f"\nProcessing image {image_name} with original aspect ratio {original_aspect_ratio_a}x{original_aspect_ratio_b} and target aspect ratio {target_aspect_ratio_a}x{target_aspect_ratio_b}"
                )
                full_path = os.path.join(
                    config.SIMPLE_FILL_HOME, SAMPLE_DATASET_PATH, image_name + ".PNG"
                )
                img = cv2.imread(str(full_path))
                img_height, img_width, _ = img.shape
                ooi_bbox = {'x': object_of_interest_x_coord,
                            'y': object_of_interest_y_coord,
                            'width': object_of_interest_width,
                            'height': object_of_interest_height}

                fill_dimensions = compute_outpaint_area_dimensions(img, target_aspect_ratio_a/target_aspect_ratio_b, ooi_bbox)

                self.assertEqual(fill_dimensions[Location.LEFT],
                                 expected_outpaint_left + outpaint_horizontal_offset_object_of_interest,
                                 f"Image {image_name}: background width on the left differs from the expected")  # add assertion here

                self.assertEqual(fill_dimensions[Location.RIGHT],
                                 expected_outpaint_right - outpaint_horizontal_offset_object_of_interest,
                                 f"Image {image_name}: background width on the right differs from the expected")

                self.assertEqual(fill_dimensions[Location.TOP],
                                 expected_outpaint_top + outpaint_offset_vertical_object_of_interest,
                                 f"Image {image_name}: background height from the top differs from the expected")

                self.assertEqual(fill_dimensions[Location.BOTTOM],
                                 expected_outpaint_bottom - outpaint_offset_vertical_object_of_interest,
                                 f"Image {image_name}: background height from the bottom differs from the expected")


# Edge Detection Test Parameters
#
#   image_name (str): the file name of the current image under test
#
#
#   object_of_interest_x_coord (int): the x coordinate of the upper left corner of the object of interest
#
#   object_of_interest_y_coord (int): the y coordinate of the upper left corner of the object of interest
#
#   object_of_interest_width (int): the width of the object of interest
#
#   object_of_interest_height (int): the height of the object of interest
#
#   gradient_background_width_left (int): the width of the gradient background on the left of the object of interest
#
#   gradient_background_width_right (int): the width of the gradient background on the right of the object of interest
#
#   gradient_background_height_top (int): the height of the gradient background on the top of the object of interest
#
#   gradient_background_height_bottom (int): the height of the gradient background on the bottom of
#   the object of interest

data_param_list_background_edge_detection = [
    # (
    #     "<image_file_name>",
    #     343,  # object_of_interest_x_coord
    #     200,  # object_of_interest_y_coord
    #     1078,  # object_of_interest_width
    #     3199,  # object_of_interest_height
    #     192,  # gradient_background_width_left #TODO: THE SHADOW ALTERS THE VALUE, the real value is 349
    #     501,  # gradient_background_width_right
    #     197,  # gradient_background_height_top
    #     27,  # gradient_background_height_bottom
    # ),
]


class EdgeDetectorTestCase(unittest.TestCase):

    def test_largest_background_dimensions(self):
        for (
            image_name,
            object_of_interest_x_coord,
            object_of_interest_y_coord,
            object_of_interest_width,
            object_of_interest_height,
            expected_background_width_left,
            expected_background_width_right,
            expected_background_width_top,
            expected_background_width_bottom,
        ) in data_param_list_background_edge_detection:
            with self.subTest(
                image_name=image_name,
                object_of_interest_x_coord=object_of_interest_x_coord,
                object_of_interest_y_coord=object_of_interest_y_coord,
                object_of_interest_width=object_of_interest_width,
                object_of_interest_height=object_of_interest_height,
                expected_background_width_left=expected_background_width_left,
                expected_background_width_right=expected_background_width_right,
                expected_background_width_top=expected_background_width_top,
                expected_background_width_bottom=expected_background_width_bottom,
            ):

                print(
                    f"\nProcessing image {image_name}"
                )
                full_path = os.path.join(
                    config.SIMPLE_FILL_HOME, SAMPLE_DATASET_PATH, image_name + ".PNG"
                )
                img = cv2.imread(str(full_path))
                img_height, img_width, _ = img.shape
                background_dimensions = find_largest_smooth_background_dimensions(img)

                delta = 5
                if abs(object_of_interest_x_coord - background_dimensions[Location.LEFT]) > delta:
                    warnings.warn(
                        f"Image {image_name}: background width on the left differs too much from the OOI x coordinate! "
                        f"Check for shadow interference!")

                if abs(object_of_interest_y_coord - background_dimensions[Location.TOP]) > delta:
                    warnings.warn(
                        f"Image {image_name}: background height from the top differs too much from the OOI y coordinate! "
                        f"Check for shadow interference!")

                if abs(img_width - object_of_interest_x_coord - object_of_interest_width -
                       background_dimensions[Location.RIGHT]) > delta:
                    warnings.warn(
                        f"Image {image_name}: background width on the right differs too much from the OOI width! "
                        f"Check for shadow interference!")

                if abs(img_height - object_of_interest_y_coord - object_of_interest_height -
                       background_dimensions[Location.BOTTOM]) > delta:
                    warnings.warn(
                        f"Image {image_name}: background height on the bottom differs too much from the OOI height! "
                        f"Check for shadow interference!")

                self.assertEqual(background_dimensions[Location.LEFT], expected_background_width_left,
                                 f"Image {image_name}: background width on the left differs from the expected")  # add assertion here

                self.assertEqual(background_dimensions[Location.RIGHT], expected_background_width_right,
                                 f"Image {image_name}: background width on the right differs from the expected")

                self.assertEqual(background_dimensions[Location.TOP], expected_background_width_top,
                                 f"Image {image_name}: background height from the top differs from the expected")

                self.assertEqual(background_dimensions[Location.BOTTOM], expected_background_width_bottom,
                                 f"Image {image_name}: background height from the bottom differs from the expected")


def check_window_present_with_delta(window, windows, delta):
    for w in windows:
        if abs(w[0] - window[0]) < delta and abs(w[1] - window[1]) < delta:
            return True
    return False


def check_regions_present_in_test_data(regions, test_data_dict, delta_dict):
    def regions_by_column(regions):
        regions_by_col = {}
        seq_col_indices = list()
        for region in regions:
            col = region.col_idx
            if col not in regions_by_col:
                regions_by_col[col] = []
            regions_by_col[col].append(region)
            if col not in seq_col_indices:
                seq_col_indices.append(col)
        return regions_by_col, seq_col_indices

    reg_by_col, col_indices = regions_by_column(regions)
    found_main_grad_transition_region = dict()

    for col, col_regions in reg_by_col.items():
        seq_col_num = col_indices.index(col) + 1
        found_main_grad_transition_region[col] = False
        for region in col_regions:
            if region.is_shadow:
                expected_grad = test_data_dict[f'expected_gradient_level_shadow_region_{seq_col_num}']
                expected_cov = test_data_dict[f'expected_coverage_shadow_region_{seq_col_num}']
                expected_start = test_data_dict[f'expected_start_shadow_region_{seq_col_num}']
                expected_end = test_data_dict[f'expected_end_shadow_region_{seq_col_num}']
                if abs(region.intensity_gradient - expected_grad) > delta_dict['intensity_gradient']:
                    return False
                if abs(region.region_coverage - expected_cov) > delta_dict['region_coverage']:
                    return False
                if abs(region.row_idx_visible_start - expected_start) > delta_dict['start_row']:
                    return False
                if abs(region.row_idx_visible_end - expected_end) > delta_dict['end_row']:
                    return False
            else:
                expected_grad = test_data_dict[f'expected_gradient_level_transition_region_{seq_col_num}']
                expected_cov = test_data_dict[f'expected_coverage_gradient_transition_region_{seq_col_num}']
                expected_start = test_data_dict[f'expected_start_gradient_transition_region_{seq_col_num}']
                expected_end = test_data_dict[f'expected_end_gradient_transition_region_{seq_col_num}']

                if abs(region.intensity_gradient - expected_grad) > delta_dict['intensity_gradient']:
                    return False
                if abs(region.region_coverage - expected_cov) > delta_dict['region_coverage']:
                    return False
                if abs(region.row_idx_visible_start - expected_start) > delta_dict['start_row']:
                    return False
                if abs(region.row_idx_visible_end - expected_end) > delta_dict['end_row']:
                    return False
                found_main_grad_transition_region[col] = True
    return True


delta_dict = {'intensity_gradient': 0.05, 'region_coverage': 0.05, 'start_row': 3, 'end_row': 3}


background_region_data_fields = [
    'image_name',
    'gradient_background_width_left',
    'gradient_background_width_right',
    'gradient_background_width_top',
    'gradient_background_width_bottom',
    'col_index_of_test_column_1',
    'expected_gradient_level_transition_region_1',
    'expected_coverage_gradient_transition_region_1',
    'expected_start_gradient_transition_region_1',
    'expected_end_gradient_transition_region_1',
    'expected_gradient_level_shadow_region_1',
    'expected_coverage_shadow_region_1',
    'expected_start_shadow_region_1',
    'expected_end_shadow_region_1',
    'col_index_of_test_column_2',
    'expected_gradient_level_transition_region_2',
    'expected_coverage_gradient_transition_region_2',
    'expected_start_gradient_transition_region_2',
    'expected_end_gradient_transition_region_2',
    'expected_gradient_level_shadow_region_2',
    'expected_coverage_shadow_region_2',
    'expected_start_shadow_region_2',
    'expected_end_shadow_region_2',
    'col_index_of_test_column_3',
    'expected_gradient_level_transition_region_3',
    'expected_coverage_gradient_transition_region_3',
    'expected_start_gradient_transition_region_3',
    'expected_end_gradient_transition_region_3',
    'expected_gradient_level_shadow_region_3',
    'expected_coverage_shadow_region_3',
    'expected_start_shadow_region_3',
    'expected_end_shadow_region_3',
]

background_region_data = namedtuple('background_region_data', background_region_data_fields)

data_param_list_background_region = [
    # background_region_data(
    #     "<image_file_name>",
    #     890,  # gradient_background_width_left
    #     930,  # gradient_background_width_right
    #     180,  # gradient_background_height_top
    #     180,  # gradient_background_height_bottom
    #     860,  # col index of the first test column                                                 #####################
    #     0.391,  # gradient level for the transition region of the first test column                #
    #     0.9,  # coverage for the transition region of the first test column                        #
    #     2828,  # row index for top of the gradient transition region of the first test column      #
    #     2885,  # row index for bottom of the gradient transition region of the first test column   # Test Column 1 Data
    #     0.964,  # gradient level for the shadow region of the first test column                    #
    #     0.9,  # coverage for the shadow region of the first test column                            #
    #     3250,  # row index for top of the shadow region of the first test column                   #
    #     3332,  # row index for bottom of the shadow region of the first test column                #
    #     660,  # col index of the second test column                                                #####################
    #     0.444,  # gradient level for the transition region of the second test column               #
    #     0.9,  # coverage for the transition region of the second test column                       #
    #     2826,  # row index for top of the gradient transition region of the second test column     #
    #     2846,  # row index for bottom of the gradient transition region of the second test column  # Test Column 2 Data
    #     0.44,  # gradient level for the shadow region of the second test column                    #
    #     0.9,  # coverage for the shadow region of the second test column                           #
    #     3229,  # row index for top of the shadow region of the second test column                  #
    #     3321,  # row index for bottom of the shadow region of the second test column               #
    #     460,  # col index of the third test column                                                 #####################
    #     0.4,  # gradient level for the transition region of the third test column                  #
    #     0.9,  # coverage for the transition region of the third test column                        #
    #     2868,  # row index for start of the gradient transition region of the third test column    #
    #     2912,  # row index for end of the gradient transition region of the third test column      # Test Column 3 Data
    #     0.354,  # gradient level for the shadow region of the third test column                    #
    #     0.9,  # coverage for the shadow region of the third test column                            #
    #     3204,  # row index for start of the shadow region of the third test column                 #
    #     3281,  # row index for end of the shadow region of the third test column                   #
    # ),
]


class ShadowDetectorTestCase(unittest.TestCase):
    def setUp(self):
        self.shadow_detector = ShadowDetector(add_only_shadow_regions=False)

    def test_find_vertical_gradient_regions(self):
        for data_item in data_param_list_background_region:
            with self.subTest(data_item=data_item):
                print(
                    f"\nProcessing image {data_item.image_name}"
                )
                full_path = os.path.join(
                    config.SIMPLE_FILL_HOME, SAMPLE_DATASET_PATH, data_item.image_name + ".PNG"
                )
                img = cv2.imread(str(full_path))
                gray_background = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_height, img_width, _ = img.shape
                background_columns = [data_item.col_index_of_test_column_1, data_item.col_index_of_test_column_2, data_item.col_index_of_test_column_3]
                regions = self.shadow_detector.find_vertical_gradient_regions(gray_background, background_columns)

                self.assertTrue(len(regions) > 0)
                self.assertTrue(check_regions_present_in_test_data(regions, data_item._asdict(), delta_dict))


shadow_region_removed_data_fields = [
    'image_name',
    'gradient_background_width_left',
    'gradient_background_width_right',
    'gradient_background_width_top',
    'gradient_background_width_bottom',
    'col_index_of_test_column_1',
    'expected_gradient_level_transition_region_1',
    'expected_coverage_gradient_transition_region_1',
    'expected_start_gradient_transition_region_1',
    'expected_end_gradient_transition_region_1',
    'expected_gradient_level_transition_region_2',
    'expected_coverage_transition_region_2',
    'expected_start_transition_region_2',
    'expected_end_transition_region_2',
    'col_index_of_test_column_2',
    'expected_gradient_level_transition_region_3',
    'expected_coverage_gradient_transition_region_3',
    'expected_start_gradient_transition_region_3',
    'expected_end_gradient_transition_region_3',
    'expected_gradient_level_transition_region_4',
    'expected_coverage_transition_region_4',
    'expected_start_transition_region_4',
    'expected_end_transition_region_4',
    'col_index_of_test_column_3',
    'expected_gradient_level_transition_region_5',
    'expected_coverage_gradient_transition_region_5',
    'expected_start_gradient_transition_region_5',
    'expected_end_gradient_transition_region_5',
    'expected_gradient_level_transition_region_6',
    'expected_coverage_transition_region_6',
    'expected_start_transition_region_6',
    'expected_end_transition_region_6',
]

shadow_region_removed_data = namedtuple('shadow_region_removed_data', shadow_region_removed_data_fields)

data_param_list_shadow_region_removed = [
    # shadow_region_removed_data(
    #     "<image_file_name>",
    #     890,  # gradient_background_width_left
    #     930,  # gradient_background_width_right
    #     180,  # gradient_background_height_top
    #     180,  # gradient_background_height_bottom
    #     860,  # col index of the first test column                                                 #####################
    #     34.0,  # gradient level for the first transition region of the first test column           #
    #     0.9,  # coverage for the first transition region of the first test column                  #
    #     2806,  # row index for top of the first transition region of the first test column         #
    #     2948,  # row index for bottom of the first transition region of the first test column      # Test Column 1 Data
    #     98.0,  # gradient level for the second transition region of the first test column          #
    #     0.9,  # coverage for the second transition region of the first test column                 #
    #     3248,  # row index for top of the second transition region of the first test column        #
    #     3332,  # row index for bottom of the second transition region of the first test column     #
    #     660,  # col index of the second test column                                                #####################
    #     34.0,  # gradient level for the first transition region of the second test column          #
    #     0.9,  # coverage for the first transition region of the second test column                 #
    #     2806,  # row index for top of the first transition region of the second test column        #
    #     2948,  # row index for bottom of the first transition region of the second test column     # Test Column 2 Data
    #     44.0,  # gradient level for the second transition region of the second test column         #
    #     0.9,  # coverage for the second transition region of the second test column                #
    #     3226,  # row index for top of the second transition region of the second test column       #
    #     3319,  # row index for bottom of the second transition region of the second test column    #
    #     460,  # col index of the third test column                                                 #####################
    #     34,  # gradient level for the first transition region of the third test column             #
    #     0.9,  # coverage for the first transition region of the third test column                  #
    #     2780,  # row index for start of the first transition region of the third test column       #
    #     2942,  # row index for end of the first transition region of the third test column         # Test Column 3 Data
    #     29.0,  # gradient level for the second transition region of the third test column          #
    #     0.9,  # coverage for the second transition region of the third test column                 #
    #     3205,  # row index for start of the second transition region of the third test column      #
    #     3287,  # row index for end of the second transition region of the third test column        #
    # ),
]


class ShadowRemoverTestCase(unittest.TestCase):
    def setUp(self):
        self.shadow_remover = ShadowRemover()

    def test_remove_shadows_all_locations(self):
        for data_item in data_param_list_shadow_region_removed:
            with self.subTest(data_item=data_item):
                print(
                    f"\nProcessing image {data_item.image_name}"
                )
                full_path = os.path.join(
                    config.SMOOTH_GRADIENT_OUTPAINTER_HOME, SAMPLE_DATASET_PATH, data_item.image_name + ".PNG"
                )
                img = cv2.imread(str(full_path))
                gradient_backgrounds = {Location.LEFT: data_item.gradient_background_width_left,
                                        Location.RIGHT: data_item.gradient_background_width_right,
                                        Location.TOP: data_item.gradient_background_width_top,
                                        Location.BOTTOM: data_item.gradient_background_width_bottom}
                shadow_regions = self.shadow_remover.detect_shadow(img, gradient_backgrounds)
                self.shadow_remover.remove_shadows_all_locations(img, shadow_regions)
                self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
