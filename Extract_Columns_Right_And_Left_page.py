# -*- coding: utf-8 -*-
"""Enhanced City Directory Column Extractor

Extracts text columns from scanned city directory pages with improved page detection,
column separation, and image saving capabilities.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import interpolation as inter
import os
from pathlib import Path
import glob

class StructuralLineDetector:
    """Detects and visualizes structural lines in document images with easy line retrieval."""

    def __init__(self, blur_kernel=3, canny_low=100, canny_high=200,
                 hough_threshold=100, max_line_gap=10, line_thickness=4):
        self.blur_kernel = blur_kernel
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.max_line_gap = max_line_gap
        self.line_thickness = line_thickness

        # Store detected lines for easy retrieval
        self.detected_lines = {}  # Structure: {direction: {image_index: lines}}
        self.image_shapes = []    # Store image dimensions for reference

    def load_images(self, img_paths):
        """Load multiple images in grayscale."""
        images = []
        valid_paths = []
        
        for path in img_paths:
            try:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    valid_paths.append(path)
                    print(f"Successfully loaded: {Path(path).name}")
                else:
                    print(f"Warning: Could not load image: {Path(path).name}")
            except Exception as e:
                print(f"Error loading {Path(path).name}: {e}")
        
        self.image_shapes = [img.shape for img in images]
        return images, valid_paths

    def preprocess_image(self, img):
        """Apply blur and edge detection to image."""
        blurred = cv2.medianBlur(img.copy(), self.blur_kernel)
        return cv2.Canny(blurred, self.canny_low, self.canny_high)

    def enhance_lines(self, canny_img, direction='horizontal'):
        """Enhance structural lines using morphological operations."""
        if direction == 'horizontal':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
        else:  # vertical
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))

        return cv2.dilate(canny_img, kernel, iterations=1)

    def detect_lines(self, enhanced_img, img_shape, direction='horizontal'):
        """Detect lines using Hough transform."""
        if direction == 'horizontal':
            min_length = img_shape[1] // 4  # width / 4
        else:  # vertical
            min_length = int(img_shape[0] * 0.5)  # height * 0.5

        lines = cv2.HoughLinesP(
            enhanced_img, 1, np.pi / 180, self.hough_threshold,
            minLineLength=min_length, maxLineGap=self.max_line_gap
        )

        # Merge nearby lines to avoid duplicates
        if lines is not None:
            lines = self.merge_nearby_lines(lines, direction)

        return lines

    def merge_nearby_lines(self, lines, direction='vertical', merge_distance=10):
        """Merge lines that are very close to each other."""
        if lines is None or len(lines) == 0:
            return []

        # Convert to list of tuples for easier handling
        line_coords = [line[0] for line in lines]

        if direction == 'vertical':
            # Sort by x coordinate (average of x1 and x2)
            line_coords.sort(key=lambda line: (line[0] + line[2]) / 2)
            merged = [line_coords[0]]

            for current_line in line_coords[1:]:
                last_merged = merged[-1]
                # Compare average x positions
                if abs((current_line[0] + current_line[2])/2 - (last_merged[0] + last_merged[2])/2) <= merge_distance:
                    # Merge: keep the longer line or average the coordinates
                    merged[-1] = current_line  # Simple: just take the current one
                else:
                    merged.append(current_line)
        else:  # horizontal
            # Sort by y coordinate
            line_coords.sort(key=lambda line: (line[1] + line[3]) / 2)
            merged = [line_coords[0]]

            for current_line in line_coords[1:]:
                last_merged = merged[-1]
                if abs((current_line[1] + current_line[3])/2 - (last_merged[1] + last_merged[3])/2) <= merge_distance:
                    merged[-1] = current_line
                else:
                    merged.append(current_line)

        return np.array([[line] for line in merged])  # Convert back to expected format

    def draw_lines(self, img, lines, color=(255, 0, 0)):
        """Draw detected lines on image."""
        result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result, (x1, y1), (x2, y2), color, self.line_thickness)

        return result

    def process_images(self, img_paths, directions=['horizontal', 'vertical']):
        """Process multiple images for structural line detection."""
        images, valid_paths = self.load_images(img_paths)
        
        if not images:
            print("No valid images found to process!")
            return {}
            
        results = {}

        # Initialize detected_lines storage
        for direction in ['horizontal', 'vertical']:
            self.detected_lines[direction] = {}

        # Display original images
        self._plot_images(images, 'Original Images')

        # Handle 'both' direction option
        if 'both' in directions:
            directions = ['horizontal', 'vertical', 'both']

        for direction in directions:
            processed_images = []
            line_images = []

            for i, img in enumerate(images):
                if direction == 'both':
                    # Process both horizontal and vertical lines together
                    canny = self.preprocess_image(img)

                    # Enhance both directions
                    enhanced_hor = self.enhance_lines(canny, 'horizontal')
                    enhanced_vert = self.enhance_lines(canny, 'vertical')

                    # Detect lines in both directions
                    lines_hor = self.detect_lines(enhanced_hor, img.shape, 'horizontal')
                    lines_vert = self.detect_lines(enhanced_vert, img.shape, 'vertical')

                    # Store detected lines
                    self.detected_lines['horizontal'][i] = lines_hor if lines_hor is not None else []
                    self.detected_lines['vertical'][i] = lines_vert if lines_vert is not None else []

                    # Combine enhanced images
                    enhanced_combined = cv2.max(enhanced_hor, enhanced_vert)

                    # Draw lines from both directions on same image
                    line_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    if lines_hor is not None:
                        for line in lines_hor:
                            x1, y1, x2, y2 = line[0]
                            cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), self.line_thickness)  # Red for horizontal
                    if lines_vert is not None:
                        for line in lines_vert:
                            x1, y1, x2, y2 = line[0]
                            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), self.line_thickness)  # Green for vertical

                    processed_images.append(enhanced_combined)
                    line_images.append(line_img)

                else:
                    # Process single direction (original logic)
                    canny = self.preprocess_image(img)
                    enhanced = self.enhance_lines(canny, direction)

                    # Detect and draw lines
                    detected_lines = self.detect_lines(enhanced, img.shape, direction)

                    # Store detected lines
                    self.detected_lines[direction][i] = detected_lines if detected_lines is not None else []

                    line_img = self.draw_lines(img, detected_lines)

                    processed_images.append(enhanced)
                    line_images.append(line_img)

            # Store results
            results[direction] = {
                'enhanced': processed_images,
                'with_lines': line_images
            }

            # Uncomment BELOW TO Display Separator Results - DEBUGGING
            #self._plot_images(line_images, f'Detected {direction.title()} Lines')

        return results, valid_paths

    def get_lines(self, direction='both', image_index=None):
        """
        Retrieve detected lines with optional filtering.

        Args:
            direction: 'horizontal', 'vertical', or 'both'
            image_index: Specific image index (0-based), or None for all images

        Returns:
            Dictionary with line data
        """
        if direction == 'both':
            result = {}
            for dir_type in ['horizontal', 'vertical']:
                if image_index is not None:
                    result[dir_type] = self.detected_lines.get(dir_type, {}).get(image_index, [])
                else:
                    result[dir_type] = self.detected_lines.get(dir_type, {})
            return result
        else:
            if image_index is not None:
                return self.detected_lines.get(direction, {}).get(image_index, [])
            else:
                return self.detected_lines.get(direction, {})

    def _plot_images(self, images, title, figsize=(20, 10)):
        """Helper method to plot multiple images."""
        n_images = len(images)
        plt.figure(figsize=figsize)

        for i, img in enumerate(images):
            plt.subplot(1, n_images, i + 1)
            plt.title(f'{title} - Image {i + 1}')

            # Handle both grayscale and color images
            if len(img.shape) == 3:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(img, cmap='gray')

            plt.axis('on')

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(10)
        plt.close()  # Close the plot to avoid blocking the script


class CityDirectoryExtractor:
    """Enhanced extractor for city directory pages with page type detection and column separation."""
    
    def __init__(self, detector, output_dir="extracted_columns"):
        self.detector = detector
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Thresholds for page classification and column extraction
        # Left Page 
        self.LEFT_PAGE_AD_THRESHOLD = 0.2  # 20% from left for advertisement detection
        self.RIGHT_PAGE_AD_THRESHOLD = 0.75  # 75% from left for right page detection
        self.LEFT_PAGE_TOP_THRESHOLD = 0.33     # 33% from top
        self.BOTTOM_THRESHOLD_LEFT = 0.85   # 85% for left column
        self.BOTTOM_THRESHOLD_RIGHT = 0.75  # 75% for right column
        
    def detect_page_type(self, image_index):
        """
        Detect if page is left or right based on vertical line positions.
        Left pages typically have advertisements on the left side.
        """
        img_width = self.detector.image_shapes[image_index][1]
        midpoint_x = int(img_width / 2) # Midpoint of the image width 
        
        vertical_lines = self.detector.get_lines('vertical', image_index)

        vert_separator = self._find_closest_vertical_line(vertical_lines, midpoint_x)

        if vert_separator is None or vert_separator > midpoint_x:
            return 'left'
        else:
            return 'right' 
    
    def find_column_separators(self, image_index, page_type):
        """Find column separator coordinates based on page type."""
        img_height, img_width = self.detector.image_shapes[image_index]
        vertical_lines = self.detector.get_lines('vertical', image_index)
        horizontal_lines = self.detector.get_lines('horizontal', image_index)
        
        separators = {}
        
        if page_type == 'left':
            separators = self._find_left_page_separators(
                img_height, img_width, vertical_lines, horizontal_lines
            )
        else:  # right page
            separators = self._find_right_page_separators(
                img_height, img_width, vertical_lines, horizontal_lines
            )
            
        return separators
    
    def _find_left_page_separators(self, img_height, img_width, vertical_lines, horizontal_lines):
        """Find separators for left-hand pages."""
        mid_x = img_width // 2
        left_threshold = int(img_width * self.LEFT_PAGE_AD_THRESHOLD)
        
        # Find main column separator (closest to middle)
        x_sep = self._find_closest_vertical_line(vertical_lines, mid_x)
        
        # Find advertisement separator (leftmost significant line)
        x_sep_left = self._find_leftmost_vertical_line(vertical_lines, left_threshold)
        
        # Find horizontal separators
        y_separators = self._find_left_page_horizontal_separators(horizontal_lines, img_height)
        
        return {
            'x_sep': x_sep,
            'x_sep_left': x_sep_left,
            'x_sep_right': img_width,  # Right separator is just to the right of main separator
            'y_sep_top_left': max(0, y_separators['top_left'] - 10),
            'y_sep_bottom_left': min(img_height, y_separators['bottom_left'] + 10),
            'y_sep_top_right': max(0, y_separators['top_right'] - 10),
            'y_sep_bottom_right': min(img_height, y_separators['bottom_right'] + 10),
        }
    
    def _find_right_page_separators(self, img_height, img_width, vertical_lines, horizontal_lines):
        """Find separators for right-hand pages."""
        mid_x = img_width // 2
        right_threshold = int(img_width * self.RIGHT_PAGE_AD_THRESHOLD)
        
        # For right pages, main separator is still near middle
        x_sep = self._find_closest_vertical_line(vertical_lines, mid_x)
        
        # Find advertisement separator (rightmost significant line)
        x_sep_right = self._find_rightmost_vertical_line(vertical_lines, right_threshold)
        
        # Find horizontal separators
        y_separators = self._find_right_page_horizontal_separators(horizontal_lines, img_height)
        
        return {
            'x_sep': x_sep,
            'x_sep_left': 0,
            'x_sep_right': x_sep_right,
            'y_sep_top_left': max(0, y_separators['top_left'] - 10),
            'y_sep_bottom_left': min(img_height, y_separators['bottom_left'] + 10),
            'y_sep_top_right': max(0, y_separators['top_right'] - 10),
            'y_sep_bottom_right': min(img_height, y_separators['bottom_right'] + 15),
        }
    
    def _find_closest_vertical_line(self, vertical_lines, target_x):
        """Find vertical line closest to target x-coordinate."""
        if vertical_lines is None or len(vertical_lines) == 0:
            return target_x
            
        min_distance = float('inf')
        best_x = target_x
        
        for line in vertical_lines:
            x = line[0][0]
            distance = abs(x - target_x)
            if distance < min_distance:
                min_distance = distance
                best_x = x
                
        return best_x
    
    def _find_leftmost_vertical_line(self, vertical_lines, threshold):
        """Find leftmost vertical line within threshold. This is used for left page ad removal"""
        if vertical_lines is None or len(vertical_lines) == 0:
            return 0
            
        best_x = 0
        for line in vertical_lines:
            x = line[0][0]
            if x < threshold and x > best_x:
                best_x = x
        return best_x
    
    def _find_rightmost_vertical_line(self, vertical_lines, threshold):
        """Find rightmost vertical line within threshold. This is used for left page ad removal"""
        if vertical_lines is None or len(vertical_lines) == 0:
            return threshold # Default to threshold if no lines found
            
        best_x = self.detector.image_shapes[0][1]  # Start with image width
        for line in vertical_lines:
            x = line[0][0]
            if x > threshold and x < best_x:
                best_x = x
        return best_x
    
    def _find_left_page_horizontal_separators(self, horizontal_lines, img_height):
        """Find top and bottom horizontal separators for left pages."""
        if horizontal_lines is None or len(horizontal_lines) == 0:
            # Return default values if no horizontal lines found
            return {
                'top_left': 0,
                'bottom_left': int(0.75 * img_height),
                'top_right': 0,
                'bottom_right': int(0.75 * img_height)
            }
            
        left_page_top_threshold = int(img_height * self.LEFT_PAGE_TOP_THRESHOLD)
        bottom_threshold_left = int(img_height * self.BOTTOM_THRESHOLD_LEFT)
        bottom_threshold_right = int(img_height * self.BOTTOM_THRESHOLD_RIGHT)
        
        y_sep_top_left = 0
        y_sep_bottom_left = int(0.75 * img_height)
        y_sep_bottom_right = int(0.75 * img_height)
        
        # Process horizontal lines
        for line in horizontal_lines:
            y = line[0][1]
            
            # Top separator for left column
            if y < left_page_top_threshold and y > y_sep_top_left:
                y_sep_top_left = y
                
            # Bottom separator for left column
            if y > bottom_threshold_left and y > y_sep_bottom_left and y < img_height:
                y_sep_bottom_left = y
                
            # Bottom separator for right column
            if y > bottom_threshold_right and y < y_sep_bottom_right:
                y_sep_bottom_right = y
        
        # Top right separator (assume 2nd horizontal line if available)
        y_sep_top_right = horizontal_lines[1][0][1] if len(horizontal_lines) > 1 else y_sep_top_left
        
        return {
            'top_left': y_sep_top_left,
            'bottom_left': y_sep_bottom_left,
            'top_right': y_sep_top_right,
            'bottom_right': y_sep_bottom_right
        }
    
    def _find_right_page_horizontal_separators(self, horizontal_lines, img_height):
        """Find top and bottom horizontal separators for right pages."""
        if horizontal_lines is None or len(horizontal_lines) == 0:
            # Return default values if no horizontal lines found
            return {
                'top_left': 0,
                'bottom_left': int(0.75 * img_height),
                'top_right': 0,
                'bottom_right': int(0.75 * img_height)
            }
            
        right_page_top_threshold = int(img_height * self.LEFT_PAGE_TOP_THRESHOLD)
        bottom_threshold_left = int(img_height * self.BOTTOM_THRESHOLD_RIGHT)
        bottom_threshold_right = int(img_height * self.BOTTOM_THRESHOLD_LEFT)
        
        y_sep_top_right = 0
        y_sep_bottom_left = int(0.75 * img_height)
        y_sep_bottom_right = int(0.75 * img_height)
        
        # Process horizontal lines
        for line in horizontal_lines:
            y = line[0][1]
            
            # Top separator for right column
            if y < right_page_top_threshold and y > y_sep_top_right:
                y_sep_top_right = y
                
            # Bottom separator for right column
            if y > bottom_threshold_left and y > y_sep_bottom_right and y < img_height:
                y_sep_bottom_right = y
                
            # Bottom separator for left column
            if y > bottom_threshold_left and y < y_sep_bottom_left:
                y_sep_bottom_left = y
        
        # Top left separator (assume 2nd horizontal line if available)
        y_sep_top_left = horizontal_lines[1][0][1] if len(horizontal_lines) > 1 else y_sep_top_right
        
        return {
            'top_left': y_sep_top_left,
            'bottom_left': y_sep_bottom_left,
            'top_right': y_sep_top_right,
            'bottom_right': y_sep_bottom_right
        }
        
    def extract_columns(self, image_paths):
        """Extract columns from all images and save as separate files."""
        extracted_data = {}
        
        for i, img_path in enumerate(image_paths):
            print(f"Processing image {i}: {Path(img_path).name}")
            
            # Detect page type
            page_type = self.detect_page_type(i)
            print(f"  Detected page type: {page_type}")
            
            # Find separators
            separators = self.find_column_separators(i, page_type)
            
            # Load original image
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                print(f"  Warning: Could not load image for extraction: {Path(img_path).name}")
                continue
            
            # Extract columns
            left_column = image[
                separators['y_sep_top_left']:separators['y_sep_bottom_left'],
                separators['x_sep_left']:separators['x_sep']
            ]
            
            right_column = image[
                separators['y_sep_top_right']:separators['y_sep_bottom_right'],
                separators['x_sep']:separators['x_sep_right']
            ]
            
            # Store extracted data
            extracted_data[i] = {
                'page_type': page_type,
                'left_column': left_column,
                'right_column': right_column,
                'separators': separators,
                'original_path': img_path
            }
            
            # Save column images
            self._save_column_images(i, img_path, left_column, right_column)
            
            print(f"  Left column size: {left_column.shape}")
            print(f"  Right column size: {right_column.shape}")
            
        return extracted_data
        
    def _save_column_images(self, image_index, original_path, left_column, right_column):
        """Save extracted columns as separate image files."""
        base_name = Path(original_path).stem
        
        # Save left column
        left_path = self.output_dir / f"{base_name}_left_col.jpg"
        cv2.imwrite(str(left_path), left_column)
        
        # Save right column
        right_path = self.output_dir / f"{base_name}_right_col.jpg"
        cv2.imwrite(str(right_path), right_column)
        
        print(f"  Saved: {left_path.name} and {right_path.name}")
    
    def visualize_extraction(self, extracted_data, max_images=3):
        """Visualize the extracted columns."""
        n_images = min(len(extracted_data), max_images)
        
        plt.figure(figsize=(20, 6 * n_images))
        
        for i in range(n_images):
            data = extracted_data[i]
            
            # Left column
            plt.subplot(n_images, 2, i * 2 + 1)
            plt.imshow(data['left_column'], cmap='gray')
            plt.title(f"Image {i} - Left Column ({data['page_type']} page)")
            plt.axis('off')
            
            # Right column
            plt.subplot(n_images, 2, i * 2 + 2)
            plt.imshow(data['right_column'], cmap='gray')
            plt.title(f"Image {i} - Right Column ({data['page_type']} page)")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(6)
        plt.close()  # Close the plot to avoid blocking the script


def get_image_files(directory_path, extensions=['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']):
    """
    Get all image files from a directory.
    
    Args:
        directory_path: Path to the directory containing images
        extensions: List of valid image file extensions
    
    Returns:
        List of full paths to image files
    """
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Error: Directory does not exist: {directory_path}")
        return []
    
    image_files = []
    for ext in extensions:
        # Search for files with this extension (case-insensitive)
        pattern = str(directory / f"*{ext}")
        image_files.extend(glob.glob(pattern))
        
        # Also search for uppercase extensions
        pattern_upper = str(directory / f"*{ext.upper()}")
        image_files.extend(glob.glob(pattern_upper))
    
    # Remove duplicates and sort
    image_files = sorted(list(set(image_files)))
    
    print(f"Found {len(image_files)} image files in {directory_path}")
    for img_file in image_files:
        print(f"  - {Path(img_file).name}")
    
    return image_files


# Example usage
if __name__ == "__main__":
    # Directory containing all images to process (relative to current script location)
    IMAGE_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_images')
    
    # Get all image files from the directory
    image_paths = get_image_files(IMAGE_DIRECTORY)
    
    if not image_paths:
        print("No image files found in the specified directory!")
        exit(1)
    
    print(f"\nProcessing {len(image_paths)} images from directory...")
    
    # Initialize detector
    detector = StructuralLineDetector(
        blur_kernel=3,
        canny_low=100,
        canny_high=200,
        hough_threshold=1200,
        max_line_gap=10,
        line_thickness=4
    )
    
    # Process images to detect lines
    results, valid_paths = detector.process_images(image_paths, directions=['both'])
    
    if not valid_paths:
        print("No valid images were processed!")
        exit(1)
    
    # Initialize extractor
    extractor = CityDirectoryExtractor(detector, output_dir="extracted_columns")
    
    # Extract columns and save as images
    extracted_data = extractor.extract_columns(valid_paths)
    
    # Visualize results (showing first few images)
    if extracted_data:
        extractor.visualize_extraction(extracted_data)
        print(f"\nExtraction complete! Column images saved in: {extractor.output_dir}")
        print(f"Successfully processed {len(extracted_data)} images")
    else:
        print("No images were successfully processed for column extraction!")