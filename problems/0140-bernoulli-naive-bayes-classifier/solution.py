import numpy as np

class NaiveBayes():
    def __init__(self, smoothing=1.0):
        # Initialize smoothing
        self.smoothing = smoothing

    def forward(self, X, y):
        # Fit model to binary features X and labels y
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes)

        # Store class prior probabilities
        self.class_priors = {}

        # Store conditional probabilities for each feature
        self.feature_probs = {}

        for c in self.classes:
            # Samples belonging to class c
            X_c = X[y == c]
            n_c = X_c.shape[0]

            # P(y = c)
            self.class_priors[c] = n_c / n_samples

            # P(x_j = 1 | y = c), with Laplace smoothing
            feature_counts = np.sum(X_c, axis=0)

            self.feature_probs[c] = (
                feature_counts + self.smoothing
            ) / (
                n_c + 2 * self.smoothing
            )

    def predict(self, X):
        # Predict class labels for test set X
        predictions = []

        for x in X:
            class_scores = []

            for c in self.classes:
                probs = self.feature_probs[c]

                # Log prior probability
                score = np.log(self.class_priors[c])

                # Bernoulli Naive Bayes log likelihood
                score += np.sum(
                    x * np.log(probs) +
                    (1 - x) * np.log(1 - probs)
                )

                class_scores.append(score)

            # Select class with highest posterior probability
            predictions.append(self.classes[np.argmax(class_scores)])

        return np.array(predictions)