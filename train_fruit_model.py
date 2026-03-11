# ============================================================================
# FRUIT QUALITY DETECTION - CNN + XGBoost
# Optimized for CPU with Batch Processing
# Dataset: 6000 images, 10 classes (5 fruits × 2 conditions)
# ============================================================================

import os
import warnings

# Silence TensorFlow + Keras warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

# ============================================================================
# CONFIGURATION
# ============================================================================
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
DATASET_PATH = 'fruit_vegetable_quality'
BATCH_SIZE = 32  # Process 32 images at once 
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Class names 
CLASS_NAMES = [
    'fresh_apple', 'fresh_banana', 'fresh_mango', 'fresh_orange', 'fresh_strawberry',
    'rotten_apple', 'rotten_banana', 'rotten_mango', 'rotten_orange', 'rotten_strawberry']

print("="*70)
print("FRUIT QUALITY DETECTION - HYBRID CNN + XGBoost")
print("Optimized for CPU with Batch Processing")
print("="*70)

# ============================================================================
# STEP 1: LOAD DATASET PATHS
# ============================================================================
def load_dataset(dataset_path):
    """
    Load all image paths and labels from hierarchical folder structure.
    
    fruit_quality/
    ├── fresh/
    │   ├── apple/
    │   ├── banana/
    │   ├── mango/
    │   ├── orange/
    │   └── strawberry/
    │   
    └── rotten/
        ├── apple/
        ├── banana/
        ├── mango/
        ├── orange/
        └── strawberry/

    """
    print("\n" + "="*70)
    print("STEP 1: LOADING DATASET")
    print("="*70)
    
    image_paths = []
    labels = []
    
    conditions = ['fresh', 'rotten']
    fruits = ['apple', 'banana', 'mango', 'orange', 'strawberry']  
    
    class_count = {}
    label_idx = 0
    
    for condition in conditions:
        for fruit in fruits:
            class_name = f"{condition}_{fruit}"
            folder_path = os.path.join(dataset_path, condition, fruit)
            
            if not os.path.exists(folder_path):
                print(f"⚠️  Warning: {folder_path} not found, skipping...")
                continue
            
            count = 0
            for img_file in os.listdir(folder_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(folder_path, img_file)
                    image_paths.append(img_path)
                    labels.append(label_idx)
                    count += 1
            
            class_count[class_name] = count
            label_idx += 1
    
    print(f"\n📊 Dataset Summary:")
    print(f"Total images: {len(image_paths)}")
    print(f"Total classes: {len(class_count)}")
    print(f"\n📁 Images per class:")
    for class_name, count in class_count.items():
        print(f"  {class_name:20s}: {count:4d} images")
    
    return image_paths, labels

# ============================================================================
# STEP 2: CREATE FEATURE EXTRACTOR
# ============================================================================
def create_feature_extractor():
    """
    Load pretrained MobileNetV2 as feature extractor.
    """
    print("\n" + "="*70)
    print("STEP 2: LOADING MOBILENETV2 FEATURE EXTRACTOR")
    print("="*70)
    
    print("Loading pretrained MobileNetV2 (ImageNet weights)...")
    
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,  # Remove classification layer
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
        pooling='avg'  # Global average pooling
    )
    
    # Freeze all layers
    base_model.trainable = False
    
    print(f"✅ MobileNetV2 loaded successfully!")
    print(f"   Output shape: {base_model.output_shape}")
    print(f"   Feature vector size: 1280")
    print(f"   Total parameters: {base_model.count_params():,}")
    
    return base_model

# ============================================================================
# STEP 3: EXTRACT FEATURES (BATCH PROCESSING - OPTIMIZED FOR CPU)
# ============================================================================
def extract_features_batch(image_paths, labels, feature_extractor, batch_size=32):
    """
    Extract features from all images using batch processing.
    This is 2-3× faster than processing one image at a time!
    
    Args:
        image_paths: List of image file paths
        labels: List of corresponding labels
        feature_extractor: MobileNetV2 model
        batch_size: Number of images to process at once (32 optimal for CPU)
    
    Returns:
        X: Feature matrix (num_images, 1280)
        y: Label array (num_images,)
    """
    print("\n" + "="*70)
    print("STEP 3: EXTRACTING FEATURES (BATCH PROCESSING)")
    print("="*70)
    
    features = []
    valid_labels = []
    failed_images = []
    
    total_images = len(image_paths)
    total_batches = (total_images + batch_size - 1) // batch_size
    
    print(f"\n⚙️  Configuration:")
    print(f"   Total images: {total_images}")
    print(f"   Batch size: {batch_size}")
    print(f"   Total batches: {total_batches}")
    print(f"   Estimated time: 12-15 minutes")
    print(f"\n🔄 Processing batches...")
    print("-" * 70)
    
    start_time = time.time()
    
    for batch_idx in range(0, total_images, batch_size):
        batch_start_time = time.time()
        
        # Get current batch
        batch_end = min(batch_idx + batch_size, total_images)
        batch_paths = image_paths[batch_idx:batch_end]
        batch_labels = labels[batch_idx:batch_end]
        
        batch_images = []
        batch_valid_labels = []
        
        # Load all images in this batch
        for img_path, label in zip(batch_paths, batch_labels):
            try:
                # Load and preprocess image
                img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
                img_array = image.img_to_array(img)
                img_array = preprocess_input(img_array)
                
                batch_images.append(img_array)
                batch_valid_labels.append(label)
                
            except Exception as e:
                failed_images.append((img_path, str(e)))
                continue
        
        # Process entire batch at once (KEY OPTIMIZATION!)
        if batch_images:
            batch_array = np.array(batch_images)  # Shape: (batch_size, 224, 224, 3)
            batch_features = feature_extractor.predict(batch_array, verbose=0)
            
            features.extend(batch_features)
            valid_labels.extend(batch_valid_labels)
        
        # Progress reporting
        current_batch = (batch_idx // batch_size) + 1
        processed = min(batch_end, total_images)
        percentage = (processed / total_images) * 100
        batch_time = time.time() - batch_start_time
        elapsed_time = time.time() - start_time
        
        # Estimate remaining time
        if current_batch > 0:
            avg_time_per_batch = elapsed_time / current_batch
            remaining_batches = total_batches - current_batch
            estimated_remaining = avg_time_per_batch * remaining_batches
            eta = str(timedelta(seconds=int(estimated_remaining)))
        else:
            eta = "calculating..."
        
        # Print progress every batch
        print(f"Batch {current_batch:3d}/{total_batches} | "
              f"Images: {processed:4d}/{total_images} ({percentage:5.1f}%) | "
              f"Batch time: {batch_time:4.1f}s | "
              f"ETA: {eta}")
    
    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(valid_labels)
    
    total_time = time.time() - start_time
    
    print("-" * 70)
    print(f"\n✅ Feature extraction complete!")
    print(f"   Successfully processed: {len(valid_labels):,} images")
    print(f"   Failed: {len(failed_images)} images")
    print(f"   Total time: {str(timedelta(seconds=int(total_time)))}")
    print(f"   Average speed: {len(valid_labels)/total_time:.1f} images/second")
    print(f"\n📊 Output shapes:")
    print(f"   Feature matrix (X): {X.shape}")
    print(f"   Label array (y): {y.shape}")
    
    if failed_images:
        print(f"\n⚠️  Failed images:")
        for img_path, error in failed_images[:5]:  # Show first 5
            print(f"   {img_path}: {error}")
        if len(failed_images) > 5:
            print(f"   ... and {len(failed_images) - 5} more")
    
    return X, y

# ============================================================================
# SAVE AND LOAD FEATURES (TIME SAVER!)
# ============================================================================
def save_features(X, y, prefix='fruit_features'):
    """Save extracted features to disk"""
    np.save(f'{prefix}_X.npy', X)
    np.save(f'{prefix}_y.npy', y)
    print(f"\n💾 Features saved:")
    print(f"   {prefix}_X.npy ({X.nbytes / 1024 / 1024:.1f} MB)")
    print(f"   {prefix}_y.npy ({y.nbytes / 1024:.1f} KB)")

def load_features(prefix='fruit_features'):
    """Load pre-extracted features from disk"""
    print("\n📂 Loading pre-extracted features from disk...")
    X = np.load(f'{prefix}_X.npy')
    y = np.load(f'{prefix}_y.npy')
    print(f"✅ Features loaded successfully!")
    print(f"   Feature matrix: {X.shape}")
    print(f"   Labels: {y.shape}")
    return X, y

def features_exist(prefix='fruit_features'):
    """Check if feature files exist"""
    return (os.path.exists(f'{prefix}_X.npy') and 
            os.path.exists(f'{prefix}_y.npy'))

# ============================================================================
# STEP 4: TRAIN-TEST SPLIT
# ============================================================================
def split_data(X, y):
    """Split data into training and testing sets with stratification"""
    print("\n" + "="*70)
    print("STEP 4: TRAIN-TEST SPLIT")
    print("="*70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # Maintain class distribution
    )
    
    print(f"\n📊 Split summary:")
    print(f"   Training set: {X_train.shape[0]:,} samples ({(1-TEST_SIZE)*100:.0f}%)")
    print(f"   Testing set:  {X_test.shape[0]:,} samples ({TEST_SIZE*100:.0f}%)")
    print(f"\n📋 Class distribution in training set:")
    unique, counts = np.unique(y_train, return_counts=True)
    for class_idx, count in zip(unique, counts):
        if class_idx < len(CLASS_NAMES):
            print(f"   {CLASS_NAMES[class_idx]:20s}: {count:4d} samples")
    
    print(f"\n📋 Class distribution in testing set:")
    unique, counts = np.unique(y_test, return_counts=True)
    for class_idx, count in zip(unique, counts):
        if class_idx < len(CLASS_NAMES):
            print(f"   {CLASS_NAMES[class_idx]:20s}: {count:4d} samples")
    
    return X_train, X_test, y_train, y_test

# ============================================================================
# STEP 5: TRAIN XGBOOST (MULTICLASS)
# ============================================================================
def train_xgboost(X_train, y_train, num_classes=10):
    """Train XGBoost classifier for multiclass classification"""
    print("\n" + "="*70)
    print("STEP 5: TRAINING XGBOOST CLASSIFIER")
    print("="*70)
    
    print(f"\n⚙️  Model configuration:")
    print(f"   Algorithm: XGBoost")
    print(f"   Task: Multiclass classification")
    print(f"   Number of classes: {num_classes}")
    print(f"   Number of trees: 100")
    print(f"   Max depth: 6")
    print(f"   Learning rate: 0.1")
    
    model = xgb.XGBClassifier(
        objective='multi:softmax',      # Multiclass classification
        num_class=num_classes,          # Number of classes
        max_depth=6,                    # Maximum tree depth
        learning_rate=0.1,              # Step size
        n_estimators=100,               # Number of trees
        eval_metric='mlogloss',         # Multiclass log loss
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,                      # Use all CPU cores
        verbosity=1
    )
    
    print(f"\n🔄 Training in progress...")
    start_time = time.time()
    
    model.fit(X_train, y_train, verbose=True)
    
    train_time = time.time() - start_time
    
    print(f"\n✅ Training complete!")
    print(f"   Training time: {str(timedelta(seconds=int(train_time)))}")
    
    return model

# ============================================================================
# STEP 6: EVALUATE MODEL
# ============================================================================
def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("\n" + "="*70)
    print("STEP 6: MODEL EVALUATION")
    print("="*70)
    
    print("\n🔄 Making predictions on test set...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Overall accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n📊 OVERALL ACCURACY: {accuracy * 100:.2f}%")
    
    # Detailed classification report
    print(f"\n📋 Detailed Classification Report:")
    print("="*70)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)
    print(report)
    
    # Per-class accuracy
    print(f"\n📊 Per-Class Accuracy:")
    print("-"*70)
    for i, class_name in enumerate(CLASS_NAMES):
        class_mask = y_test == i
        if class_mask.sum() > 0:
            class_correct = (y_pred[class_mask] == i).sum()
            class_total = class_mask.sum()
            class_acc = class_correct / class_total
            print(f"{class_name:20s}: {class_acc*100:6.2f}% ({class_correct:3d}/{class_total:3d})")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📊 Confusion Matrix (10×10):")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={'label': 'Number of Predictions'},
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title('Confusion Matrix - Fruit Quality Detection (10 Classes)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_10class.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Confusion matrix saved as 'confusion_matrix_10class.png'")
    
    # Feature importance (top 20)
    feature_importance = model.feature_importances_
    top_20_idx = np.argsort(feature_importance)[-20:][::-1]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(20), feature_importance[top_20_idx])
    plt.yticks(range(20), [f'Feature {i}' for i in top_20_idx])
    plt.xlabel('Importance Score', fontweight='bold')
    plt.title('Top 20 Most Important Features', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"💾 Feature importance plot saved as 'feature_importance.png'")
    
    return accuracy

# ============================================================================
# STEP 7: SAVE MODELS
# ============================================================================
def save_models(xgboost_model, feature_extractor=None):
    """Save trained models"""
    print("\n" + "="*70)
    print("STEP 7: SAVING MODELS")
    print("="*70)
    
    # Save XGBoost
    xgboost_model.save_model('fruit_quality_xgboost.json')
    print(f"✅ XGBoost model saved: 'fruit_quality_xgboost.json'")
    
    # Save MobileNetV2 (optional)
    if feature_extractor is not None:
        feature_extractor.save('fruit_mobilenetv2_feature_extractor.h5')
        print(f"✅ Feature extractor saved: 'fruit_mobilenetv2_feature_extractor.h5'")

# ============================================================================
# PREDICT SINGLE IMAGE
# ============================================================================
def predict_single_image(img_path, feature_extractor, xgboost_model):
    """
    Predict the class of a single image
    
    Args:
        img_path: Path to image file
        feature_extractor: MobileNetV2 model
        xgboost_model: Trained XGBoost model
    
    Returns:
        predicted_class: Class name
        confidence: Prediction confidence (0-1)
    """
    print(f"\n{'='*70}")
    print("SINGLE IMAGE PREDICTION")
    print(f"{'='*70}")
    print(f"Image: {img_path}")
    
    # Load and preprocess
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Extract features
    features = feature_extractor.predict(img_array, verbose=0)
    
    # Predict
    prediction = xgboost_model.predict(features)[0]
    prediction_proba = xgboost_model.predict_proba(features)[0]
    
    predicted_class = CLASS_NAMES[prediction]
    confidence = prediction_proba[prediction]
    
    print(f"\n🎯 PREDICTION: {predicted_class.upper()}")
    print(f"📊 Confidence: {confidence * 100:.2f}%")
    print(f"\n📋 All class probabilities:")
    for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, prediction_proba)):
        bar = '█' * int(prob * 50)
        print(f"   {class_name:20s}: {prob*100:5.2f}% {bar}")
    
    return predicted_class, confidence

# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    """Main execution function"""
    
    print(f"\n🕒 Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    overall_start = time.time()
    
    # Check if features already extracted
    if features_exist('fruit_features'):
        print("\n" + "="*70)
        print("✅ PRE-EXTRACTED FEATURES FOUND!")
        print("="*70)
        print("Loading features from disk (saves 12-15 minutes)...")
        
        X, y = load_features('fruit_features')
        feature_extractor = None  # Don't need to load CNN
        
    else:
        print("\n" + "="*70)
        print("⚠️  No saved features found - Starting fresh extraction")
        print("="*70)
        
        # Step 1: Load dataset
        image_paths, labels = load_dataset(DATASET_PATH)
        
        # Step 2: Create feature extractor
        feature_extractor = create_feature_extractor()
        
        # Step 3: Extract features with batching
        X, y = extract_features_batch(
            image_paths, 
            labels, 
            feature_extractor, 
            batch_size=BATCH_SIZE
        )
        
        # Save features for future use
        save_features(X, y, 'fruit_features')
        print("\n💡 TIP: Next time you run this, features will load in 2 seconds!")
    
    # Step 4: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 5: Train XGBoost
    model = train_xgboost(X_train, y_train, num_classes=len(CLASS_NAMES))
    
    # Step 6: Evaluate
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Step 7: Save models
    save_models(model, feature_extractor)
    
    # Summary
    total_time = time.time() - overall_start
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"✅ Total execution time: {str(timedelta(seconds=int(total_time)))}")
    print(f"✅ Final accuracy: {accuracy * 100:.2f}%")
    print(f"✅ Model saved: 'fruit_quality_xgboost.json'")
    print(f"✅ Features saved: 'fruit_features_X.npy', 'fruit_features_y.npy'")
    print(f"\n🕒 End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n{'='*70}")
    print("NEXT STEPS:")
    print("="*70)
    print("1. Check 'confusion_matrix_10class.png' for detailed results")
    print("2. Check 'feature_importance.png' for feature analysis")
    print("3. Use predict_single_image() to test on new images")
    print("4. Re-run this script to experiment with parameters (loads in 2-3 min!)")
    print("="*70)

# ============================================================================
# LOAD AND PREDICT (for deployment)
# ============================================================================
def load_and_predict(image_path):
    """
    Load saved models and predict on a new image
    Use this after training is complete
    """
    print("Loading saved models...")
    
    # Load feature extractor
    feature_extractor = tf.keras.models.load_model('fruit_mobilenetv2_feature_extractor.h5')
    
    # Load XGBoost
    xgboost_model = xgb.XGBClassifier()
    xgboost_model.load_model('fruit_quality_xgboost.json')
    
    # Predict
    predicted_class, confidence = predict_single_image(
        image_path, 
        feature_extractor, 
        xgboost_model
    )
    
    return predicted_class, confidence

# ============================================================================
# RUN THE PROGRAM
# ============================================================================
if __name__ == "__main__":
    main()
    
    # Example: Predict on a single image (uncomment to use)
load_and_predict('test_images/test_image3.png')
