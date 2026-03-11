# ============================================================
# FRUIT QUALITY DETECTION – PREDICTION WITH RATING
# CNN (MobileNetV2) + XGBoost + Quality Rating (0-5 Stars)
# ============================================================
import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import xgboost as xgb
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------- CONFIG ----------------
IMG_HEIGHT = 224
IMG_WIDTH = 224
FEATURE_EXTRACTOR_PATH = "fruit_mobilenetv2_feature_extractor.h5"
XGB_MODEL_PATH = "fruit_quality_xgboost.json"

CLASS_NAMES = [
    'fresh_apple', 'fresh_banana', 'fresh_mango', 'fresh_orange', 'fresh_strawberry',
    'rotten_apple', 'rotten_banana', 'rotten_mango', 'rotten_orange', 'rotten_strawberry'
]

# Rating thresholds for FRESH fruits (confidence-based)
RATING_THRESHOLDS = {
    5: 0.95,  # Excellent: 95%+ confidence
    4: 0.85,  # Very Good: 85-95%
    3: 0.70,  # Good: 70-85%
    2: 0.55,  # Fair: 55-70%
    1: 0.40   # Poor: 40-55%
}

# ---------------------------------------

def load_models():
    """Load trained CNN feature extractor and XGBoost model"""
    print("🔄 Loading models...")
    cnn_model = tf.keras.models.load_model(FEATURE_EXTRACTOR_PATH)
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(XGB_MODEL_PATH)
    print("✅ Models loaded successfully\n")
    return cnn_model, xgb_model

def extract_features(img_path, cnn_model):
    """Extract CNN features from a single image"""
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = cnn_model.predict(img_array, verbose=0)
    return features

def calculate_rating(predicted_class, confidence):
    """
    Calculate quality rating (0-5 stars)
    
    Logic:
    - Rotten fruit: 0 stars (rejected, do not consume)
    - Fresh fruit: 1-5 stars based on confidence
        - 5 stars: Excellent quality (95%+ confidence)
        - 4 stars: Very Good quality (85-95%)
        - 3 stars: Good quality (70-85%)
        - 2 stars: Fair quality (55-70%)
        - 1 star:  Poor quality (40-55%)
        - 0 stars: Below 40% confidence or rotten
    
    Returns:
        rating (int): 0-5 stars
        rating_label (str): Description of the rating
    """
    # Check if fruit is rotten
    if predicted_class.startswith('rotten'):
        return 0, "Rejected - Do Not Consume"
    
    # For fresh fruits, rate based on confidence
    if confidence >= RATING_THRESHOLDS[5]:
        return 5, "Excellent Quality"
    elif confidence >= RATING_THRESHOLDS[4]:
        return 4, "Very Good Quality"
    elif confidence >= RATING_THRESHOLDS[3]:
        return 3, "Good Quality"
    elif confidence >= RATING_THRESHOLDS[2]:
        return 2, "Fair Quality"
    elif confidence >= RATING_THRESHOLDS[1]:
        return 1, "Poor Quality - Inspect Carefully"
    else:
        return 0, "Uncertain - Manual Inspection Required"

def get_star_display(rating):
    """Convert rating to star display"""
    filled_stars = "★" * rating
    empty_stars = "☆" * (5 - rating)
    return filled_stars + empty_stars

def get_emoji_by_rating(rating):
    """Get emoji representation for rating"""
    emoji_map = {
        5: "🌟",
        4: "✨",
        3: "👍",
        2: "⚠️",
        1: "❗",
        0: "❌"
    }
    return emoji_map.get(rating, "❓")

def predict_image(img_path, cnn_model, xgb_model):
    """Predict class, confidence, and quality rating"""
    features = extract_features(img_path, cnn_model)
    probs = xgb_model.predict_proba(features)[0]
    
    best_idx = np.argmax(probs)
    predicted_class = CLASS_NAMES[best_idx]
    confidence = probs[best_idx]
    
    # Calculate quality rating
    rating, rating_label = calculate_rating(predicted_class, confidence)
    
    return predicted_class, confidence, probs, rating, rating_label

def main():
    if len(sys.argv) != 2:
        print("❌ Usage:")
        print("   python predict_with_rating.py path/to/image.jpg")
        sys.exit(1)
    
    img_path = sys.argv[1]
    
    if not os.path.exists(img_path):
        print(f"❌ ERROR: Image not found → {img_path}")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("🍎 FRUIT QUALITY DETECTION – PREDICTION WITH RATING")
    print("=" * 70)
    print(f"Image: {img_path}\n")
    
    # Load models and predict
    cnn_model, xgb_model = load_models()
    predicted_class, confidence, probs, rating, rating_label = predict_image(
        img_path, cnn_model, xgb_model
    )
    
    # Display prediction
    print(f"🎯 PREDICTION: {predicted_class.upper()}")
    print(f"📊 Confidence: {confidence * 100:.2f}%")
    
    # Display rating
    print("\n" + "─" * 70)
    print("🌟 QUALITY RATING")
    print("─" * 70)
    
    emoji = get_emoji_by_rating(rating)
    stars = get_star_display(rating)
    
    print(f"{emoji}  Rating: {rating}/5  {stars}")
    print(f"Status: {rating_label}")
    
    # Interpretation guide
    if rating == 0:
        if predicted_class.startswith('rotten'):
            print("\n⚠️  ALERT: This fruit is ROTTEN and should NOT be consumed.")
            print("   Recommendation: Discard immediately.")
        else:
            print("\n⚠️  ALERT: Low confidence prediction.")
            print("   Recommendation: Manual inspection required.")
    elif rating == 1:
        print("\n⚠️  WARNING: Poor quality detected.")
        print("   Recommendation: Inspect carefully before consuming.")
    elif rating == 2:
        print("\nℹ️  Fair quality - consumable but not optimal.")
        print("   Recommendation: Use soon or in cooked dishes.")
    elif rating == 3:
        print("\n✓ Good quality - suitable for consumption.")
        print("   Recommendation: Safe to eat, consume within normal timeframe.")
    elif rating == 4:
        print("\n✓ Very good quality - fresh and healthy.")
        print("   Recommendation: Excellent for eating or selling.")
    elif rating == 5:
        print("\n✓ Excellent quality - premium grade!")
        print("   Recommendation: Perfect for direct consumption or premium sale.")
    
    # Display all probabilities
    print("\n📋 All class probabilities:")
    print("─" * 70)
    for cls, p in sorted(zip(CLASS_NAMES, probs), key=lambda x: x[1], reverse=True):
        bar = "█" * int(p * 40)
        print(f"{cls:20s}: {p * 100:6.2f}% {bar}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
