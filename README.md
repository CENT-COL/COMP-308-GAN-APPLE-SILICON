# MNIST ACGAN Browser Demo - Apple Silicon Compatible

## 🚀 Overview

This interactive browser-based ACGAN (Auxiliary Classifier Generative Adversarial Network) demo was specifically created to provide an accessible, educational experience for training and exploring GANs on Apple Silicon (M1/M2/M3) computers. Unlike traditional Node.js implementations that require complex dependencies, this demo runs entirely in your web browser using TensorFlow.js with WebGL GPU acceleration.

## 🎯 Why This Demo Was Created

### Educational Motivation
- **Instant Access**: Students can explore GANs immediately without installing Node.js, Python, or dealing with dependency conflicts
- **Visual Learning**: Real-time training progress with live loss graphs and generated image updates
- **Interactive Exploration**: Adjustable parameters (epochs, batch size) to understand their impact on training
- **Apple Silicon Compatibility**: Overcomes the limitation that `tfjs-node-gpu` doesn't support ARM64 architecture

### Technical Motivation
- **WebGL Acceleration**: Leverages your Mac's GPU through the browser's WebGL backend
- **Memory Optimization**: Smart tensor management and garbage collection for sustained training
- **Parameter Synchronization**: Matches the original Node.js implementation parameters for authentic results
- **Multiple Data Sources**: Support for both real MNIST data and synthetic data for different learning scenarios

## 🎓 GAN Concepts Demonstrated

### 1. **Auxiliary Classifier GAN (ACGAN) Architecture**
The demo implements a complete ACGAN with two main components:

#### Generator (`buildGenerator()` - Line ~668)
```javascript
// The generator takes two inputs:
// 1. Random latent vector (noise)
// 2. Desired digit class (0-9)
const latent = tf.input({shape: [CONFIG.latentSize]});      // Random "seed"
const imageClass = tf.input({shape: [1]});                  // Digit class

// Class embedding converts class labels to vectors
const classEmbedding = tf.layers.embedding({
    inputDim: CONFIG.numClasses,     // 10 classes (digits 0-9)
    outputDim: CONFIG.latentSize     // 100-dimensional embedding
}).apply(imageClass);

// Combine noise with class information
const h = tf.layers.multiply().apply([latent, classEmbedding]);
```

**Key Concept**: The generator learns to create realistic MNIST digits by combining random noise with specific class information, allowing controlled generation of specific digits.

#### Discriminator (`buildDiscriminator()` - Line ~718)
```javascript
// Two outputs make this an "Auxiliary Classifier"
const realnessScore = tf.layers.dense({
    units: 1,
    activation: 'sigmoid'           // Real vs Fake classification
}).apply(features);

const aux = tf.layers.dense({
    units: CONFIG.numClasses,       // Digit classification (0-9)
    activation: 'softmax'
}).apply(features);
```

**Key Concept**: Unlike basic GANs, ACGAN's discriminator performs two tasks:
1. **Real/Fake Detection**: Determines if an image is real or generated
2. **Classification**: Identifies which digit (0-9) the image represents

### 2. **Adversarial Training Process**
The training loop (`trainModel()` - Line ~812) demonstrates the classic GAN training dynamic:

#### Discriminator Training (`trainDiscriminatorStep()` - Line ~767)
```javascript
// Train on mix of real and fake images
const x = tf.concat([realImages, fakeImages], 0);
const y = tf.concat([
    tf.ones([batchSize, 1]).mul(0.95),    // "Soft" labels for real images
    tf.zeros([batchSize, 1])              // Labels for fake images
], 0);
```

**Key Concept**: The discriminator learns by seeing both real MNIST images (labeled as "real") and generator outputs (labeled as "fake"). Soft labels (0.95 instead of 1.0) help training stability.

#### Generator Training (`trainGeneratorStep()` - Line ~790)
```javascript
// "Fool" the discriminator by labeling fake images as real
const trick = tf.ones([batchSize, 1]).mul(0.95);  // Tell discriminator these are "real"
```

**Key Concept**: The generator improves by trying to fool the discriminator - it gets better when the discriminator mistakes its fake images for real ones.

### 3. **Training Dynamics Visualization**
Watch the loss curves in real-time to understand GAN training:

- **Discriminator Loss**: Should start high and decrease as it learns to distinguish real from fake
- **Generator Loss**: Fluctuates as it competes with the improving discriminator
- **Equilibrium**: Successful training reaches a balance where both networks improve together

## 🛠 How to Use the Demo

### Quick Start (Recommended for First-Time Users)
1. **Open the Demo**: Double-click `gan_demo.html` or open it in your web browser
2. **Load Pre-trained Model**: Click the blue "Load Hosted Model" button
3. **Generate Images**: Click "Generate Images" to see high-quality results immediately
4. **Explore**: Try generating multiple times to see the variety of outputs

### Full Training Experience
1. **Configure Training**:
   - **Epochs**: Use the slider to set training duration (1-200 epochs)
   - **Batch Size**: Adjust batch size (16-128) based on your system's memory
   - **Data Source**: Choose "Real MNIST Dataset" for authentic training data

2. **Start Training**: Click "Start Training" and watch the real-time progress:
   - **Loss Metrics**: Monitor discriminator and generator competition
   - **Memory Usage**: See tensor and buffer usage in real-time
   - **Generated Samples**: View improving image quality every 10 epochs

3. **Interactive Controls**:
   - **Pause/Resume**: Stop training to examine current results
   - **Generate**: Create new images at any point during training
   - **Reset**: Clear everything and start fresh with new parameters

### Advanced Features
- **Parameter Comparison**: Train with different settings and compare results
- **Model Comparison**: Compare your trained model against the pre-trained hosted model
- **Memory Monitoring**: Watch GPU memory usage and garbage collection in action

## 🔧 Technical Implementation

### Apple Silicon Optimization
```javascript
// GPU Memory optimization settings (Line ~226)
tf.ENV.set('WEBGL_DELETE_TEXTURE_THRESHOLD', 0);
tf.ENV.set('WEBGL_FLUSH_THRESHOLD', 1);
tf.ENV.set('WEBGL_PACK', true);           // Efficient texture packing
tf.ENV.set('WEBGL_LAZILY_UNPACK', true);  // On-demand unpacking
```

### Memory Management
```javascript
// Periodic garbage collection (Line ~865)
if ((batch + 1) % CONFIG.memoryOptimization.gcInterval === 0) {
    const memBefore = tf.memory();
    await tf.dispose();  // Clean up orphaned tensors
    const memAfter = tf.memory();
}
```

### Real-time Data Loading
The demo supports two data modes:
- **Real MNIST**: Attempts to load actual MNIST dataset for authentic training
- **Synthetic Data**: Falls back to generated MNIST-like patterns for consistent performance

## 📚 Educational Value

### For Students
- **Immediate Gratification**: See professional results instantly with the hosted model
- **Progressive Learning**: Watch your own model improve from random noise to recognizable digits
- **Parameter Impact**: Experiment with different epochs and batch sizes to understand their effects
- **Concept Reinforcement**: Visual feedback reinforces theoretical GAN concepts

### For Instructors
- **No Setup Required**: Students can start exploring immediately
- **Consistent Results**: WebGL backend provides reliable performance across different Mac models
- **Scalable Difficulty**: Begin with pre-trained model, progress to full training
- **Real Metrics**: Authentic loss curves and training dynamics from actual GAN implementation

## 🌟 Key Features

- **🎯 Apple Silicon Native**: Optimized for M1/M2/M3 Macs using WebGL GPU acceleration
- **📊 Real-time Metrics**: Live loss tracking and memory usage monitoring
- **🎛 Interactive Controls**: Adjustable epochs (1-200) and batch size (16-128)
- **🚀 Instant Demo**: Pre-trained model for immediate high-quality results
- **📈 Educational**: Visual learning with real-time training progress
- **💾 Smart Memory**: Optimized tensor management for sustained training
- **🔄 Flexible Data**: Support for real MNIST and synthetic data sources

## 🚀 Getting Started

Simply open `gan_demo.html` in any modern web browser on your Apple Silicon Mac. No installation, no dependencies, no configuration required!

For the best experience:
1. Use Safari, Chrome, or Firefox (WebGL 2.0 support recommended)
2. Ensure your Mac has sufficient memory (8GB+ recommended for larger training sessions)
3. Close unnecessary browser tabs to maximize available GPU memory

---

**Happy Learning!** 🎓 This demo makes advanced machine learning concepts accessible and interactive, perfect for exploring the fascinating world of Generative Adversarial Networks on your Apple Silicon Mac.
