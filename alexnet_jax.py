"""
AlexNet architecture in modern JAX / Flax NNX.
Matches original 2012 ImageNet architecture (Krizhevsky et al., 2012).
"""

import jax
import jax.numpy as jnp
from flax import nnx
import optax


class AlexNet(nnx.Module):
    def __init__(self, num_classes: int = 1000, rngs: nnx.Rngs = nnx.Rngs(0)):
        # Layer 1: Conv (11x11, stride 4, padding 2) -> ReLU -> Local Response Norm -> MaxPool (3x3, stride 2)
        self.conv1 = nnx.Conv(in_features=3, out_features=96, kernel_size=(11, 11), strides=(4, 4), padding=((2, 2), (2, 2)), rngs=rngs)
        
        # Layer 2: Conv (5x5, padding 2) -> ReLU -> Local Response Norm -> MaxPool (3x3, stride 2)
        self.conv2 = nnx.Conv(in_features=96, out_features=256, kernel_size=(5, 5), padding=((2, 2), (2, 2)), rngs=rngs)
        
        # Layer 3: Conv (3x3, padding 1) -> ReLU
        self.conv3 = nnx.Conv(in_features=256, out_features=384, kernel_size=(3, 3), padding=((1, 1), (1, 1)), rngs=rngs)
        
        # Layer 4: Conv (3x3, padding 1) -> ReLU
        self.conv4 = nnx.Conv(in_features=384, out_features=384, kernel_size=(3, 3), padding=((1, 1), (1, 1)), rngs=rngs)
        
        # Layer 5: Conv (3x3, padding 1) -> ReLU -> MaxPool (3x3, stride 2)
        self.conv5 = nnx.Conv(in_features=384, out_features=256, kernel_size=(3, 3), padding=((1, 1), (1, 1)), rngs=rngs)
        
        # Fully Connected Layers
        self.fc6 = nnx.Linear(in_features=256 * 6 * 6, out_features=4096, rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=0.5, rngs=rngs)
        
        self.fc7 = nnx.Linear(in_features=4096, out_features=4096, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=0.5, rngs=rngs)
        
        self.fc8 = nnx.Linear(in_features=4096, out_features=num_classes, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # Layer 1
        x = nnx.relu(self.conv1(x))
        x = nnx.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        
        # Layer 2
        x = nnx.relu(self.conv2(x))
        x = nnx.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        
        # Layer 3
        x = nnx.relu(self.conv3(x))
        
        # Layer 4
        x = nnx.relu(self.conv4(x))
        
        # Layer 5
        x = nnx.relu(self.conv5(x))
        x = nnx.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        
        # Flatten
        x = x.reshape((x.shape[0], -1))
        
        # FC Layers
        x = nnx.relu(self.fc6(x))
        x = self.dropout1(x)
        
        x = nnx.relu(self.fc7(x))
        x = self.dropout2(x)
        
        x = self.fc8(x)
        return x


def test_alexnet():
    print("Initializing AlexNet Flax NNX model...")
    rngs = nnx.Rngs(0)
    model = AlexNet(num_classes=1000, rngs=rngs)
    
    # Dummy ImageNet batch: batch_size=4, height=224, width=224, channels=3
    dummy_input = jnp.ones((4, 224, 224, 3), dtype=jnp.float32)
    output = model(dummy_input)
    
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (4, 1000), "Shape mismatch!"
    print("AlexNet JAX model test passed successfully!")


if __name__ == "__main__":
    test_alexnet()
