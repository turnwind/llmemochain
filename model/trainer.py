import numpy as np
from typing import List, Dict, Optional

class PersonalityWeightOptimizer:
    def __init__(
        self,
        epsilon: float = 0.05,
        alpha: float = 0.1,
        beta: float = 0.9,
        momentum: Optional[List[float]] = None
    ):
        """Initialize the personality weight optimizer.
        
        Args:
            epsilon: Minimum weight threshold (default: 0.05)
            alpha: Learning rate (default: 0.1)
            beta: Momentum coefficient (default: 0.9)
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.momentum = momentum

    def optimize_weights(
        self,
        current_weights: List[float],
        true_emotion_idx: int,
        emotion_distance: float
    ) -> List[float]:
        """Optimize personality weights based on prediction accuracy.
        
        Args:
            current_weights: Current personality weights
            true_emotion_idx: Index of the true emotion
            emotion_distance: Distance between predicted and true emotions
            
        Returns:
            Updated personality weights
        """
        # Convert weights to numpy array for easier manipulation
        #weights = np.array(list(current_weights.values()))
        weights = np.array(current_weights)
        
        # Only update if prediction is close to true emotion
        if emotion_distance < 0.3:  # Threshold can be adjusted
            # Update weight for true personality
            weights[true_emotion_idx] = weights[true_emotion_idx] + \
                self.alpha * (1 - weights[true_emotion_idx])
            
            # Update other weights
            for j in range(len(weights)):
                if j != true_emotion_idx:
                    weights[j] = weights[j] - self.alpha * weights[j]
        
        # Apply momentum if available
        if self.momentum is not None:
            weights = self.beta * self.momentum + (1 - self.beta) * weights
        self.momentum = weights.copy()
        
        # Ensure minimum threshold
        weights = np.maximum(weights, self.epsilon)
        
        # Normalize to sum to 1
        weights = weights / weights.sum()
        
        # Convert back to dictionary
        return weights.tolist()

    def reset_momentum(self):
        """Reset the momentum term."""
        self.momentum = None
