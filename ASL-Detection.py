"""
============================================================
COMPLETE ASL SIGN LANGUAGE DETECTION SYSTEM
============================================================
File: main.py
Python Version: 3.10.11

SETUP INSTRUCTIONS:
1. Create a folder named "ASL_Detection"
2. Save this file as "main.py" in that folder
3. Open terminal/command prompt in that folder
4. Run: pip install -r requirements.txt
5. Run: python main.py

PROJECT STRUCTURE (Auto-created):
ASL_Detection/
├── main.py (this file)
├── requirements.txt (create this - see below)
├── README.txt (auto-generated)
├── dataset/ (auto-created)
│   ├── A/
│   ├── B/
│   └── ...
└── models/ (auto-created)
    └── asl_model.pkl

============================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time
import sys 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ==================== AUTO-SETUP ====================
def create_project_structure():
    """Automatically create project folders and files"""
    
    # Create folders
    folders = ['dataset', 'models']
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    
    # Create requirements.txt
    requirements = """opencv-python==4.8.1.78
mediapipe==0.10.8
numpy==1.24.3
scikit-learn==1.3.2
"""
    
    if not os.path.exists('requirements.txt'):
        with open('requirements.txt', 'w') as f:
            f.write(requirements)
        print("✓ Created requirements.txt")
    
    # Create README
    readme = """
============================================================
ASL SIGN LANGUAGE DETECTION SYSTEM
============================================================

INSTALLATION:
1. Install Python 3.10.11
2. Open terminal in this folder
3. Run: pip install -r requirements.txt
4. Run: python main.py

USAGE:
1. Collect Dataset (Option 1)
   - Follow on-screen instructions
   - Press SPACE to start/stop collecting
   - Press 'n' for next letter
   
2. Train Model (Option 2)
   - Automatically trains on collected data
   - Wait for completion (~1-2 minutes)
   
3. Run Detection (Option 3)
   - Show ASL signs to camera
   - See real-time predictions!

CONTROLS:
- SPACE: Start/Stop collection
- n: Next letter
- q: Quit
- s: Screenshot (in detection mode)

SUPPORTED SIGNS:
Letters: A, B, C, D, E, F, G, H, I, L, O, K, Y
Words: Hello, Thanks, ILoveYou

TIPS:
- Use good lighting
- Plain background works best
- Keep hand centered in frame
- Make clear, distinct signs

TROUBLESHOOTING:
- Camera not working? Check permissions
- Low accuracy? Collect more varied samples
- Slow performance? Close other apps

For support, check the code comments in main.py
============================================================
"""
    
    if not os.path.exists('README.txt'):
        with open('README.txt', 'w') as f:
            f.write(readme)
        print("✓ Created README.txt")


# ==================== CONFIGURATION ====================
class Config:
    """System configuration"""
    DATASET_DIR = "dataset"
    MODEL_DIR = "models"
    MODEL_PATH = os.path.join(MODEL_DIR, "asl_model.pkl")
    SAMPLES_PER_CLASS = 100
    
    # ASL Classes (16 signs for demo)
    ASL_CLASSES = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
        'L', 'O', 'K', 'Y', 'Hello', 'Thanks', 'ILoveYou'
    ]
    
    # Camera settings
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    CAMERA_INDEX = 0  # Change to 1 if default camera doesn't work
    
    # Detection settings
    MIN_DETECTION_CONFIDENCE = 0.7
    MIN_TRACKING_CONFIDENCE = 0.5


# ==================== DATASET COLLECTOR ====================
class DatasetCollector:
    """Collects training data for ASL signs"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE
        )
        self.cap = None
        
    def extract_landmarks(self, hand_landmarks):
        """Extract 63 features (21 points × 3 coordinates)"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks)
    
    def collect_data(self):
        """Main data collection loop"""
        os.makedirs(Config.DATASET_DIR, exist_ok=True)
        
        self.cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        
        if not self.cap.isOpened():
            print("❌ Error: Cannot open camera!")
            print("Try changing CAMERA_INDEX in Config class")
            return
        
        self._print_collection_header()
        
        for class_name in Config.ASL_CLASSES:
            class_dir = os.path.join(Config.DATASET_DIR, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Check existing samples
            existing_files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
            collected_samples = len(existing_files)
            
            if collected_samples >= Config.SAMPLES_PER_CLASS:
                print(f"\n✓ '{class_name}' already has {collected_samples} samples. Skipping...")
                continue
            
            print(f"\n{'='*60}")
            print(f"📝 Collecting: '{class_name}'")
            print(f"Progress: {collected_samples}/{Config.SAMPLES_PER_CLASS}")
            print(f"{'='*60}")
            print("Position your hand and press SPACE to start...")
            
            collecting = False
            sample_count = collected_samples
            
            while sample_count < Config.SAMPLES_PER_CLASS:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Camera error!")
                    break
                
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                display_frame = frame.copy()
                self._draw_collection_ui(display_frame, class_name, sample_count, 
                                        Config.SAMPLES_PER_CLASS, collecting)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            display_frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS
                        )
                        
                        if collecting:
                            landmarks = self.extract_landmarks(hand_landmarks)
                            filename = os.path.join(class_dir, f"{class_name}_{sample_count}.npy")
                            np.save(filename, landmarks)
                            sample_count += 1
                            time.sleep(0.05)
                else:
                    if collecting:
                        cv2.putText(display_frame, "⚠ NO HAND DETECTED!", (50, 250),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                
                cv2.imshow('Dataset Collection', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    if results.multi_hand_landmarks:
                        collecting = not collecting
                        status = "▶ COLLECTING" if collecting else "⏸ PAUSED"
                        print(f"{status}: {sample_count} samples")
                    else:
                        print("⚠ Show your hand first!")
                elif key == ord('n'):
                    print(f"⏭ Skipped '{class_name}' - {sample_count} samples collected")
                    break
                elif key == ord('q'):
                    print("\n❌ Collection cancelled by user")
                    self.cap.release()
                    cv2.destroyAllWindows()
                    return
            
            print(f"✅ '{class_name}' complete: {sample_count} samples")
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n" + "="*60)
        print("🎉 DATASET COLLECTION COMPLETED!")
        print("="*60)
    
    def _draw_collection_ui(self, frame, class_name, current, total, collecting):
        """Draw collection interface"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 220), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)[:]
        
        # Title with emoji
        title = f"Collecting: {class_name}"
        cv2.putText(frame, title, (30, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        
        # Progress
        progress = f"Progress: {current}/{total} ({current*100//total}%)"
        cv2.putText(frame, progress, (30, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Progress bar
        bar_width = int((w - 60) * (current / total))
        cv2.rectangle(frame, (30, 120), (30 + bar_width, 145), (0, 255, 0), -1)
        cv2.rectangle(frame, (30, 120), (w - 30, 145), (255, 255, 255), 2)
        
        # Status
        if collecting:
            status = "▶ COLLECTING... (SPACE to pause)"
            color = (0, 255, 0)
        else:
            status = "⏸ PAUSED (SPACE to start)"
            color = (0, 165, 255)
        
        cv2.putText(frame, status, (30, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Controls
        cv2.putText(frame, "Controls: SPACE=Start/Stop | n=Next | q=Quit", (30, 210),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    def _print_collection_header(self):
        """Print collection instructions"""
        print("\n" + "="*60)
        print("📸 ASL DATASET COLLECTION")
        print("="*60)
        print(f"\n🎯 Classes to collect: {len(Config.ASL_CLASSES)}")
        print(f"📊 Samples per class: {Config.SAMPLES_PER_CLASS}")
        print("\n📋 Instructions:")
        print("  • Show the ASL sign to the camera")
        print("  • Press SPACE to start/stop collecting")
        print("  • Press 'n' to skip to next letter")
        print("  • Press 'q' to quit anytime")
        print("\n💡 Tips:")
        print("  • Use good lighting")
        print("  • Keep hand centered")
        print("  • Vary hand position slightly for better data")
        print("="*60)


# ==================== MODEL TRAINER ====================
class ModelTrainer:
    """Trains machine learning model on collected data"""
    
    def __init__(self):
        self.model = None
        self.class_names = Config.ASL_CLASSES
        
    def load_dataset(self):
        """Load all collected training data"""
        print("\n📂 Loading dataset...")
        X, y = [], []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(Config.DATASET_DIR, class_name)
            if not os.path.exists(class_dir):
                print(f"⚠ Warning: No data for '{class_name}'")
                continue
            
            files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
            print(f"  Loading {len(files):3d} samples for '{class_name}'")
            
            for file in files:
                filepath = os.path.join(class_dir, file)
                landmarks = np.load(filepath)
                X.append(landmarks)
                y.append(class_idx)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\n✓ Dataset loaded: {len(X)} samples, {len(self.class_names)} classes")
        return X, y
    
    def train(self):
        """Train the classification model"""
        print("\n" + "="*60)
        print("🤖 TRAINING MODEL")
        print("="*60)
        
        X, y = self.load_dataset()
        
        if len(X) == 0:
            print("\n❌ Error: No data found!")
            print("Please collect dataset first (Option 1)")
            return False
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📊 Training set: {len(X_train)} samples")
        print(f"📊 Testing set:  {len(X_test)} samples")
        
        # Train Random Forest
        print("\n🌲 Training Random Forest Classifier...")
        print("Please wait (this may take 1-2 minutes)...")
        
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        start_time = time.time()
        self.model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Evaluate
        print("\n🔍 Evaluating model...")
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{'='*60}")
        print(f"✅ TRAINING COMPLETED!")
        print(f"{'='*60}")
        print(f"⏱ Training time: {train_time:.2f} seconds")
        print(f"🎯 Accuracy: {accuracy*100:.2f}%")
        print(f"{'='*60}")
        
        # Detailed report
        print("\n📊 Classification Report:")
        print("-"*60)
        print(classification_report(y_test, y_pred, 
                                   target_names=self.class_names,
                                   zero_division=0))
        
        self.save_model()
        
        if accuracy < 0.7:
            print("\n⚠ Warning: Accuracy is low (<70%)")
            print("Tips to improve:")
            print("  • Collect more varied samples")
            print("  • Ensure good lighting during collection")
            print("  • Make clear, distinct signs")
        
        return True
    
    def save_model(self):
        """Save trained model to disk"""
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'class_names': self.class_names,
            'version': '1.0',
            'timestamp': time.time()
        }
        
        with open(Config.MODEL_PATH, 'wb') as f:
            pickle.dump(model_data, f)
        
        file_size = os.path.getsize(Config.MODEL_PATH) / 1024
        print(f"\n💾 Model saved: {Config.MODEL_PATH}")
        print(f"📦 File size: {file_size:.1f} KB")


# ==================== REAL-TIME DETECTOR ====================
class SignLanguageDetector:
    """Real-time ASL sign detection"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE
        )
        
        self.model = None
        self.class_names = []
        self.load_model()
        
        self.cap = None
        self.current_prediction = "None"
        self.confidence = 0.0
        self.fps = 0
        self.prev_frame_time = 0
        
        # Prediction smoothing
        self.prediction_buffer = []
        self.buffer_size = 10
        
        # Screenshot counter
        self.screenshot_count = 0
        
    def load_model(self):
        """Load trained model from disk"""
        if not os.path.exists(Config.MODEL_PATH):
            print(f"\n❌ Error: Model not found!")
            print(f"Expected location: {Config.MODEL_PATH}")
            print("Please train the model first (Option 2)")
            return False
        
        print(f"\n📥 Loading model from: {Config.MODEL_PATH}")
        try:
            with open(Config.MODEL_PATH, 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.class_names = model_data['class_names']
            
            print(f"✅ Model loaded successfully!")
            print(f"🎯 Detects {len(self.class_names)} signs: {', '.join(self.class_names)}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def extract_landmarks(self, hand_landmarks):
        """Extract features from hand"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks).reshape(1, -1)
    
    def predict(self, landmarks):
        """Predict sign from landmarks"""
        if self.model is None:
            return "No Model", 0.0
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(landmarks)[0]
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        # Smoothing: buffer recent predictions
        self.prediction_buffer.append((predicted_class, confidence))
        if len(self.prediction_buffer) > self.buffer_size:
            self.prediction_buffer.pop(0)
        
        # Use most common prediction from buffer
        if len(self.prediction_buffer) >= 5:
            recent = [p[0] for p in self.prediction_buffer[-5:]]
            most_common = max(set(recent), key=recent.count)
            avg_conf = np.mean([p[1] for p in self.prediction_buffer if p[0] == most_common])
            return self.class_names[most_common], avg_conf
        
        return self.class_names[predicted_class], confidence
    
    def calculate_fps(self):
        """Calculate frames per second"""
        current_time = time.time()
        fps = 1 / (current_time - self.prev_frame_time)
        self.prev_frame_time = current_time
        self.fps = int(fps)
    
    def draw_ui(self, frame):
        """Draw user interface on frame"""
        h, w = frame.shape[:2]
        
        # Info panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 230), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)[:]
        
        # Title
        cv2.putText(frame, "ASL Real-Time Detection", (25, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {self.fps}", (25, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Prediction
        sign_text = f"Sign: {self.current_prediction}"
        cv2.putText(frame, sign_text, (25, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Confidence
        conf_text = f"Confidence: {self.confidence*100:.1f}%"
        color = (0, 255, 0) if self.confidence > 0.7 else (0, 165, 255)
        cv2.putText(frame, conf_text, (25, 170),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Confidence bar
        bar_width = int(450 * self.confidence)
        cv2.rectangle(frame, (25, 185), (25 + bar_width, 205), (0, 255, 0), -1)
        cv2.rectangle(frame, (25, 185), (475, 205), (255, 255, 255), 2)
        
        # Controls
        cv2.putText(frame, "q: Quit | s: Screenshot", (25, 225),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def run(self):
        """Main detection loop"""
        if self.model is None:
            print("❌ Cannot run detection without model!")
            return
        
        self.cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        
        if not self.cap.isOpened():
            print("❌ Error: Cannot open camera!")
            return
        
        print("\n" + "="*60)
        print("🎥 REAL-TIME SIGN LANGUAGE DETECTION")
        print("="*60)
        print("\n📋 Instructions:")
        print("  • Show ASL signs to the camera")
        print("  • Press 's' to save screenshot")
        print("  • Press 'q' to quit")
        print("\n🚀 Starting detection...\n")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Camera error!")
                break
            
            self.calculate_fps()
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand skeleton
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Predict sign
                    landmarks = self.extract_landmarks(hand_landmarks)
                    prediction, confidence = self.predict(landmarks)
                    
                    self.current_prediction = prediction
                    self.confidence = confidence
                    
                    # Draw bounding box
                    h, w, _ = frame.shape
                    x_coords = [lm.x for lm in hand_landmarks.landmark]
                    y_coords = [lm.y for lm in hand_landmarks.landmark]
                    
                    x_min = max(0, int(min(x_coords) * w) - 30)
                    x_max = min(w, int(max(x_coords) * w) + 30)
                    y_min = max(0, int(min(y_coords) * h) - 30)
                    y_max = min(h, int(max(y_coords) * h) + 30)
                    
                    box_color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 3)
                    
                    # Label above box
                    label = f"{prediction} ({confidence*100:.0f}%)"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    
                    # Label background
                    cv2.rectangle(frame, 
                                (x_min, y_min - label_size[1] - 15),
                                (x_min + label_size[0] + 10, y_min),
                                box_color, -1)
                    
                    cv2.putText(frame, label, (x_min + 5, y_min - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            else:
                self.current_prediction = "No hand detected"
                self.confidence = 0.0
            
            # Draw UI
            frame = self.draw_ui(frame)
            
            cv2.imshow('ASL Detection - Press Q to Quit', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Stopping detection...")
                break
            elif key == ord('s'):
                # Save screenshot
                self.screenshot_count += 1
                filename = f"screenshot_{self.screenshot_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Detection stopped.")


# ==================== UTILITIES ====================
def check_dependencies():
    """Check if all required packages are installed"""
    print("\n🔍 Checking dependencies...")
    
    required = {
        'cv2': 'opencv-python',
        'mediapipe': 'mediapipe',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING!")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("\n💡 Install with:")
        print("   pip install -r requirements.txt")
        print("\nOr individually:")
        for pkg in missing:
            print(f"   pip install {pkg}")
        return False
    
    print("\n✅ All dependencies installed!")
    return True


def view_dataset_info():
    """Display dataset statistics"""
    print("\n" + "="*60)
    print("📊 DATASET INFORMATION")
    print("="*60)
    
    if not os.path.exists(Config.DATASET_DIR):
        print("\n❌ No dataset found!")
        print("Run Option 1 to collect data")
        return
    
    total_samples = 0
    class_stats = []
    
    print(f"\n{'Class':<15} {'Samples':<10} {'Status':<10}")
    print("-" * 40)
    
    for class_name in Config.ASL_CLASSES:
        class_dir = os.path.join(Config.DATASET_DIR, class_name)
        if os.path.exists(class_dir):
            files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
            count = len(files)
            total_samples += count
            
            if count >= Config.SAMPLES_PER_CLASS:
                status = "✓ Complete"
            elif count > 0:
                status = f"⚠ Partial"
            else:
                status = "✗ Empty"
            
            print(f"{class_name:<15} {count:<10} {status:<10}")
            class_stats.append((class_name, count))
        else:
            print(f"{class_name:<15} {'0':<10} {'✗ Missing':<10}")
    
    print("-" * 40)
    print(f"{'Total':<15} {total_samples:<10}")
    print(f"\n📈 Classes: {len(Config.ASL_CLASSES)}")
    print(f"📊 Target samples per class: {Config.SAMPLES_PER_CLASS}")
    print(f"🎯 Total target: {len(Config.ASL_CLASSES) * Config.SAMPLES_PER_CLASS}")
    
    complete = sum(1 for _, count in class_stats if count >= Config.SAMPLES_PER_CLASS)
    print(f"✅ Complete classes: {complete}/{len(Config.ASL_CLASSES)}")
    
    if complete == len(Config.ASL_CLASSES):
        print("\n🎉 Dataset collection is COMPLETE!")
        print("You can now train the model (Option 2)")
    else:
        remaining = len(Config.ASL_CLASSES) - complete
        print(f"\n⚠ {remaining} classes need more data")
        print("Continue collection with Option 1")
    
    print("="*60)


def print_menu():
    """Display main menu"""
    print("\n" + "="*60)
    print(" " * 15 + "🤖 ASL DETECTION SYSTEM")
    print("="*60)
    print("\n📋 Select an option:\n")
    print("  1️⃣  Collect Dataset")
    print("  2️⃣  Train Model")
    print("  3️⃣  Run Real-Time Detection")
    print("  4️⃣  View Dataset Info")
    print("  5️⃣  Check Dependencies")
    print("  6️⃣  Exit")
    print("\n" + "="*60)


def print_welcome():
    """Print welcome message"""
    print("\n" + "="*60)
    print(" " * 10 + "🤖 SIGN LANGUAGE DETECTION SYSTEM")
    print(" " * 18 + "Python 3.10.11")
    print(" " * 15 + "ASL Recognition System")
    print("="*60)
    print("\n👋 Welcome! This system helps you:")
    print("   • Collect your own ASL dataset")
    print("   • Train a custom recognition model")
    print("   • Detect signs in real-time")
    print("\n💡 First time? Follow these steps:")
    print("   1. Check dependencies (Option 5)")
    print("   2. Collect dataset (Option 1)")
    print("   3. Train model (Option 2)")
    print("   4. Run detection (Option 3)")
    print("="*60)


# ==================== MAIN APPLICATION ====================
def main():
    """Main application entry point"""
    
    # Create project structure
    create_project_structure()
    
    # Print welcome message
    print_welcome()
    
    # Main loop
    while True:
        print_menu()
        
        try:
            choice = input("\n👉 Enter your choice (1-6): ").strip()
            
            if choice == '1':
                print("\n" + "="*60)
                print("📸 DATASET COLLECTION MODE")
                print("="*60)
                collector = DatasetCollector()
                collector.collect_data()
            
            elif choice == '2':
                print("\n" + "="*60)
                print("🤖 MODEL TRAINING MODE")
                print("="*60)
                
                # Check if dataset exists
                if not os.path.exists(Config.DATASET_DIR):
                    print("\n❌ No dataset found!")
                    print("Please collect dataset first (Option 1)")
                else:
                    trainer = ModelTrainer()
                    success = trainer.train()
                    if success:
                        print("\n✅ Ready for real-time detection (Option 3)!")
            
            elif choice == '3':
                print("\n" + "="*60)
                print("🎥 REAL-TIME DETECTION MODE")
                print("="*60)
                
                # Check if model exists
                if not os.path.exists(Config.MODEL_PATH):
                    print("\n❌ No trained model found!")
                    print("Please train model first (Option 2)")
                else:
                    detector = SignLanguageDetector()
                    detector.run()
            
            elif choice == '4':
                view_dataset_info()
            
            elif choice == '5':
                check_dependencies()
            
            elif choice == '6':
                print("\n" + "="*60)
                print("👋 Thank you for using ASL Detection System!")
                print("="*60)
                print("\n💾 Your data is saved in:")
                print(f"   • Dataset: {Config.DATASET_DIR}/")
                print(f"   • Model: {Config.MODEL_PATH}")
                print("\n🚀 Run again anytime with: python main.py")
                print("="*60 + "\n")
                break
            
            else:
                print("\n❌ Invalid choice! Please enter 1-6.")
        
        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted by user")
            print("Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or report this issue.")
        
        input("\n⏎ Press Enter to continue...")


# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 If this persists, check:")
        print("   • Camera is connected")
        print("   • All dependencies installed")
        print("   • Python version 3.10.11")
        input("\nPress Enter to exit...")