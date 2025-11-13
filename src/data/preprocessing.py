"""
Data preprocessing utilities
"""

import os
import argparse
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from tqdm import tqdm
import pandas as pd


class FaceDetector:
    """Face detection and alignment"""

    def __init__(self, detector_type='mtcnn', min_confidence=0.8):
        """
        Args:
            detector_type: Type of face detector ('mtcnn', 'opencv')
            min_confidence: Minimum confidence threshold
        """
        self.detector_type = detector_type
        self.min_confidence = min_confidence

        if detector_type == 'mtcnn':
            self.detector = MTCNN()
        elif detector_type == 'opencv':
            # Load Haar Cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")

    def detect_face(self, image: np.ndarray) -> Optional[dict]:
        """
        Detect face in image

        Args:
            image: Input image as numpy array (RGB)

        Returns:
            result: Dict with bounding box and confidence, or None
        """
        if self.detector_type == 'mtcnn':
            results = self.detector.detect_faces(image)

            if len(results) == 0:
                return None

            # Get the face with highest confidence
            best_result = max(results, key=lambda x: x['confidence'])

            if best_result['confidence'] < self.min_confidence:
                return None

            return {
                'box': best_result['box'],
                'confidence': best_result['confidence'],
                'keypoints': best_result.get('keypoints', None)
            }

        elif self.detector_type == 'opencv':
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50)
            )

            if len(faces) == 0:
                return None

            # Get largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face

            return {
                'box': [x, y, w, h],
                'confidence': 1.0,  # OpenCV doesn't provide confidence
                'keypoints': None
            }

    def crop_face(
        self,
        image: np.ndarray,
        face_result: dict,
        margin: float = 0.2,
        target_size: Tuple[int, int] = (224, 224)
    ) -> Optional[np.ndarray]:
        """
        Crop and resize face region

        Args:
            image: Input image
            face_result: Face detection result
            margin: Margin around face (relative to face size)
            target_size: Target size for cropped face

        Returns:
            cropped_face: Cropped and resized face image
        """
        x, y, w, h = face_result['box']

        # Add margin
        margin_w = int(w * margin)
        margin_h = int(h * margin)

        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(image.shape[1], x + w + margin_w)
        y2 = min(image.shape[0], y + h + margin_h)

        # Crop
        face = image[y1:y2, x1:x2]

        if face.size == 0:
            return None

        # Resize
        face = cv2.resize(face, target_size, interpolation=cv2.INTER_AREA)

        return face


class ImageQualityAssessor:
    """Assess image quality"""

    @staticmethod
    def assess_quality(image: np.ndarray) -> float:
        """
        Simple quality assessment based on sharpness

        Args:
            image: Input image

        Returns:
            quality_score: Score between 0 and 1
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Compute Laplacian variance (sharpness)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Normalize to [0, 1] (empirically determined range)
        quality_score = min(1.0, laplacian_var / 500.0)

        return quality_score

    @staticmethod
    def check_brightness(image: np.ndarray) -> Tuple[float, bool]:
        """
        Check if image brightness is acceptable

        Args:
            image: Input image

        Returns:
            brightness: Average brightness value
            is_acceptable: Whether brightness is in acceptable range
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        brightness = gray.mean()

        # Acceptable range: 50-200 (out of 255)
        is_acceptable = 50 <= brightness <= 200

        return brightness, is_acceptable


def preprocess_image(
    image_path: str,
    output_path: str,
    face_detector: FaceDetector,
    target_size: Tuple[int, int] = (224, 224),
    min_quality: float = 0.3
) -> dict:
    """
    Preprocess a single image

    Args:
        image_path: Path to input image
        output_path: Path to save processed image
        face_detector: Face detector instance
        target_size: Target size for output image
        min_quality: Minimum acceptable quality score

    Returns:
        result: Dict with processing results
    """
    result = {
        'success': False,
        'face_detected': False,
        'face_confidence': 0.0,
        'quality_score': 0.0,
        'error': None
    }

    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)

        # Detect face
        face_result = face_detector.detect_face(image_np)

        if face_result is None:
            result['error'] = 'No face detected'
            return result

        result['face_detected'] = True
        result['face_confidence'] = face_result['confidence']

        # Crop face
        cropped_face = face_detector.crop_face(
            image_np,
            face_result,
            target_size=target_size
        )

        if cropped_face is None:
            result['error'] = 'Failed to crop face'
            return result

        # Assess quality
        quality_score = ImageQualityAssessor.assess_quality(cropped_face)
        result['quality_score'] = quality_score

        if quality_score < min_quality:
            result['error'] = f'Low quality: {quality_score:.2f}'
            return result

        # Check brightness
        brightness, is_acceptable = ImageQualityAssessor.check_brightness(cropped_face)

        if not is_acceptable:
            result['error'] = f'Poor brightness: {brightness:.1f}'
            # Continue anyway, but log the warning

        # Save processed image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        processed_image = Image.fromarray(cropped_face)
        processed_image.save(output_path, quality=95)

        result['success'] = True

    except Exception as e:
        result['error'] = str(e)

    return result


def process_dataset(
    input_dir: str,
    output_dir: str,
    metadata_csv: str,
    output_metadata_csv: str,
    target_size: Tuple[int, int] = (224, 224),
    min_quality: float = 0.3,
    detector_type: str = 'mtcnn'
):
    """
    Process entire dataset

    Args:
        input_dir: Input directory with raw images
        output_dir: Output directory for processed images
        metadata_csv: Path to input metadata CSV
        output_metadata_csv: Path to output metadata CSV
        target_size: Target image size
        min_quality: Minimum quality threshold
        detector_type: Face detector type
    """
    # Initialize face detector
    face_detector = FaceDetector(detector_type=detector_type)

    # Load metadata
    metadata = pd.read_csv(metadata_csv)

    # Process results
    results = []

    print(f"Processing {len(metadata)} images...")

    for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
        image_path = os.path.join(input_dir, row['image_path'])
        user_id = row['user_id']

        # Output path
        rel_path = row['image_path']
        output_path = os.path.join(output_dir, rel_path)

        # Process image
        result = preprocess_image(
            image_path,
            output_path,
            face_detector,
            target_size=target_size,
            min_quality=min_quality
        )

        # Add to results
        results.append({
            'user_id': user_id,
            'image_path': rel_path,
            'face_detected': result['face_detected'],
            'face_confidence': result['face_confidence'],
            'image_quality': result['quality_score'],
            'success': result['success'],
            'error': result.get('error', '')
        })

    # Create output metadata
    results_df = pd.DataFrame(results)

    # Filter successful images
    successful = results_df[results_df['success'] == True]

    print(f"\nProcessing complete!")
    print(f"Total images: {len(results_df)}")
    print(f"Successful: {len(successful)} ({len(successful)/len(results_df)*100:.1f}%)")
    print(f"Failed: {len(results_df) - len(successful)}")

    # Save metadata
    os.makedirs(os.path.dirname(output_metadata_csv), exist_ok=True)
    successful.to_csv(output_metadata_csv, index=False)

    # Save full results for debugging
    debug_csv = output_metadata_csv.replace('.csv', '_debug.csv')
    results_df.to_csv(debug_csv, index=False)

    print(f"\nSaved successful metadata to: {output_metadata_csv}")
    print(f"Saved debug metadata to: {debug_csv}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess dating profile images')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory with raw images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for processed images')
    parser.add_argument('--metadata_csv', type=str, required=True,
                       help='Path to input metadata CSV')
    parser.add_argument('--output_metadata_csv', type=str, required=True,
                       help='Path to output metadata CSV')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Target image size')
    parser.add_argument('--min_quality', type=float, default=0.3,
                       help='Minimum quality threshold')
    parser.add_argument('--detector', type=str, default='mtcnn',
                       choices=['mtcnn', 'opencv'],
                       help='Face detector type')

    args = parser.parse_args()

    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        metadata_csv=args.metadata_csv,
        output_metadata_csv=args.output_metadata_csv,
        target_size=(args.image_size, args.image_size),
        min_quality=args.min_quality,
        detector_type=args.detector
    )


if __name__ == "__main__":
    main()
