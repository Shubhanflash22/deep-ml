
def segment_image(image, k, max_iters=100):
    H = len(image)
    W = len(image[0])

    # Step 1: Flatten image into pixels (row-major order)
    pixels = []
    for row in image:
        for pixel in row:
            pixels.append(pixel)

    # Step 2: Initialize centroids using first k pixels
    centroids = [list(pixels[i]) for i in range(k)]

    # Step 3: Run K-Means
    for _ in range(max_iters):

        clusters = [[] for _ in range(k)]

        # Assign each pixel to its nearest centroid
        for pixel in pixels:
            distances = [
                sum((pixel[c] - centroids[j][c]) ** 2
                    for c in range(3))
                for j in range(k)
            ]

            # min() returns the lowest index in case of ties
            cluster_idx = distances.index(min(distances))
            clusters[cluster_idx].append(pixel)

        # Update centroids
        new_centroids = []

        for j in range(k):
            if len(clusters[j]) == 0:
                new_centroids.append(centroids[j][:])
            else:
                mean = [
                    sum(pixel[c] for pixel in clusters[j])
                    / len(clusters[j])
                    for c in range(3)
                ]
                new_centroids.append(mean)

        # Step 4: Check convergence (exact equality)
        if new_centroids == centroids:
            break

        centroids = new_centroids

    # Step 5: Assign pixels again using final centroids
    result = []

    for i in range(H):
        row = []
        for j in range(W):
            pixel = image[i][j]

            distances = [
                sum((pixel[c] - centroids[m][c]) ** 2
                    for c in range(3))
                for m in range(k)
            ]

            cluster_idx = distances.index(min(distances))

            # Round centroid values to 4 decimal places
            row.append([
                round(value, 4)
                for value in centroids[cluster_idx]
            ])

        result.append(row)

    return result
