# Lecture 4: Mnist Dataset

## Questions and Answers

### 1. **How is a grayscale image represented on a computer? How about a color image?**
   > For grayscale, an image is represented using a 2D array. Each element (or pixel) is represented using a value from 0 (black) to 255 (white). A colour image is represented using a 3D array, with an added dimension for colour channels. The most common representation is RGB (red, green, blue).

### 2. **How are the files and folders in the `MNIST_SAMPLE` dataset structured? Why?**
```
MNIST_SAMPLE/
│
├── train/
│   ├── 3/
│   │   ├── 0.png
│   │   ├── 1.png
│   │   ├── 2.png
│   │   ├── ...
│   │   └── (thousands of images of digit 3)
│   │
│   └── 7/
│       ├── 0.png
│       ├── 1.png
│       ├── 2.png
│       ├── ...
│       └── (thousands of images of digit 7)
│
├── valid/
│   ├── 3/
│   │   ├── 0.png
│   │   ├── 1.png
│   │   ├── ...
│   │   └── (hundreds of validation images of digit 3)
│   │
│   └── 7/
│       ├── 0.png
│       ├── 1.png
│       ├── ...
│       └── (hundreds of validation images of digit 7)
│
└── labels.csv
```

### 3. **Explain how the "pixel similarity" approach to classifying digits works.**
   > The pixel similarity approach works by first creating an "ideal" version of each digit through averaging all training examples. For classification, we take a new image and compare it pixel-by-pixel to each ideal digit. At each position, we calculate the difference between pixel values. These differences are summed across all pixels to measure total distance. The ideal digit with the smallest total distance is chosen as the classification. This method treats each image as a point in high-dimensional space where each dimension represents one pixel position.

### 4. **What is a list comprehension? Create one now that selects odd numbers from a list and doubles them.**

> A list comprehension is a concise way to create a new list by applying an expression to each item in an existing list, often with a filter condition.
>
> It has the form:
> ```
> [expression for item in iterable if condition]
> ```
>
> Example that selects odd numbers and doubles them:
> ```
> [number * 2 for number in my_list if number % 2 != 0]
> ```

### 5. **What is a "rank-3 tensor"?**
> Tensor ranks describe the number of dimensions or axes:
>   - Rank-0 tensor: scalar (single number, no dimensions)
>   - Rank-1 tensor: vector (1D array, like a list of numbers)
>   - Rank-2 tensor: matrix (2D array, like a table with rows and columns)
>   - Rank-3 tensor: 3D array (like a stack of matrices or a cube of numbers)
> In the case of the MNIST dataset the batch of greyscale images are rank-3 tensors (batch_size, height, width)


### 6. **What is the difference between tensor rank and shape? How do you get the rank from the shape?**
> Rank refers to the number of dimensions or axes a tensor has, while shape describes the size of each dimension.
> Tensor rank: The number of dimensions (aka the "order" or "degree" of the tensor)
> - Scalar: rank 0
> - Vector: rank 1
> - Matrix: rank 2
> - 3D array: rank 3

> Tensor shape: A tuple specifying the length of each dimension
> - Vector of length 5: shape (5,)
> - 3×4 matrix: shape (3, 4)
> - Batch of 32 images of size 28×28: shape (32, 28, 28)

> Getting rank from shape: The rank is simply the length of the shape tuple. For example:
> - Shape (5,) → rank 1
> - Shape (3, 4) → rank 2
> - Shape (32, 28, 28) → rank 3

### 7. **What are RMSE and L1 norm?**
> RMSE (Root Mean Square Error):
> - Formula: √(mean(squared errors))
> - Process: Calculate the difference between predicted and actual values; square those differences; take the mean of those squared differences and; take the square root of that mean
> - In math notation: RMSE = √(1/n · Σ(yi - ŷi)²)

> L1 norm (Mean Absolute Error):
> - Formula: mean(|errors|)
> - Process: Calculate the difference between predicted and actual values; take the absolute value of those differences (make them positive) and; take the mean of those absolute differences
> - In math notation: L1 = 1/n · Σ|yi - ŷi|

### 8. **How can you apply a calculation on thousands of numbers at once, many thousands of times faster than a Python loop?**
> Using vectorised operations through NumPy arrays or PyTorch tensors, which leverage broadcasting and are implemented in low-level languages like C/C++.
> 
> Think of vectorized operations like this: Instead of cutting 1,000 apples one-by-one (Python loop), you place all apples on a special cutting board with 1,000 helpers who cut all apples simultaneously (vectorised operation).
> 
> When you write ```result = my_array * 2```, you're telling the computer to multiply ALL numbers by 2 at once, not one at a time.
> 
> This works because:
> - Broadcasting automatically handles different-shaped arrays
> - Low-level implementation runs optimized C/CUDA code behind the scenes
> - Parallel processing uses specialised hardware to perform many calculations simultaneously
>
> This approach is essential for deep learning, as it's what makes training neural networks feasible rather than taking years to complete.


### 9. **Create a 3×3 tensor or array containing the numbers from 1 to 9. Double it. Select the bottom-right four numbers.**
> A 3x3 tensor/array...
> ```
> [1, 2, 3]
> [4, 5, 6]
> [7, 8, 9]
> ```
> Double it...
> ```
> [2, 4, 6]
> [8, 10, 12]
> [14, 16, 18]
> ```
> The bottom-right four numbers would mean the 2x2 square (submatrix) in the corner
> ```
> [10, 12]
> [16, 18]
> ```

### 10. **What is broadcasting?**
> Broadcasting is a clever trick that lets you perform operations between arrays of different shapes without having to manually resize them.
>
> When we have 100 images of 28×28 pixels and one ideal digit image of 28×28 pixels, the shapes are:
> - Batch of images: (100, 28, 28)
> - Ideal digit: (28, 28)
>
> These aren't the same shape - the batch has an extra dimension for the 100 different images.
>
> Without broadcasting, you'd need to resize the ideal digit to shape (100, 28, 28) by making 100 copies of it. But broadcasting handles this automatically.
>
> When you write:
> ```pythonCopyresult = batch_images - ideal_digit```
> 
> Broadcasting virtually expands the (28, 28) ideal digit to (100, 28, 28) by repeating it 100 times, but without actually using extra memory. It's as if the computer is smart enough to say, "I see you want to subtract this single template from each of your 100 images, so I'll apply the same template to each one."

### 11. **Are metrics generally calculated using the training set, or the validation set? Why?**

> Metrics are calculated on the validation set because:
> - We want to measure how well our model generalizes to data it hasn't seen during training
> - Using the training set would give an overly optimistic view of model performance (since the model has already learned from that data)
> - The validation set serves as a proxy for how the model will perform on real-world data
>
> This is a fundamental principle in machine learning - we always evaluate model performance on data that wasn't used for training to get an honest assessment of how well it will work on new data.

### 12. **What is SGD?**
> SGD stands for Stochastic Gradient Descent. It's an optimisation algorithm that:
> - Takes the derivative (gradient) of the loss function with respect to model parameters
> - Updates those parameters in the opposite direction of the gradient to minimize the loss
> - Uses randomly selected subsets of data (mini-batches) instead of the entire dataset for each update, making it 'stochastic'
> 
> The 'stochastic' part is crucial - it means we use random samples rather than the whole dataset for each step. This makes training faster and introduces helpful randomness that can avoid getting stuck in local minima.
>
> The goal is to iteratively follow the gradient downhill until we reach a minimum where the derivative approaches zero, which represents the optimal model parameters that minimize the loss function.
>
> SGD optimises neural network digit recognition by iteratively adjusting weights based on prediction errors. For each batch of images, SGD:
> - Calculates how wrong our predictions are (loss)
> - Determines how each pixel weight should change to reduce errors (gradients)
> - Updates all weights slightly in their optimal directions
>
> This process gradually sculpts the weights to highlight important pixel patterns - giving more weight to distinctive features like the top line of a "7" or curves of a "3". Through thousands of small adjustments, SGD finds the optimal weight configuration that best distinguishes between digits.
>
> From a 28x28 pixel image, it's learning a complex decision boundary in 784-dimensional space that separates different digits. What makes SGD powerful is its ability to simultaneously adjust all 784 weights to minimize errors across thousands of training examples, finding patterns our brains cannot consciously perceive in such high-dimensional data.

### 13. **Why does SGD use mini-batches?**
   - Answer

### 14. **What are the seven steps in SGD for machine learning?**
   - Answer

### 15. **How do we initialize the weights in a model?**
   - Answer

### 16. **What is "loss"?**
   - Answer

### 17. **Why can't we always use a high learning rate?**
   - Answer

### 18. **What is a "gradient"?**
   - Answer

### 19. **Do you need to know how to calculate gradients yourself?**
   - Answer

### 20. **Why can't we use accuracy as a loss function?**
   - Answer

### 21. **Draw the sigmoid function. What is special about its shape?**
   - Answer

### 22. **What is the difference between a loss function and a metric?**
   - Answer

### 23. **What is the function to calculate new weights using a learning rate?**
   - Answer

### 24. **What does the `DataLoader` class do?**
   - Answer

### 18. **Write pseudocode showing the basic steps taken in each epoch for SGD.**
   - Answer

### 25. **Create a function that, if passed two arguments `[1,2,3,4]` and `'abcd'`, returns `[(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]`. What is special about that output data structure?**
   - Answer

### 26. **What are the "bias" parameters in a neural network? Why do we need them?**
   - Answer

### 27. **What does the `@` operator do in Python?**
   - Answer

### 28. **What does the `backward` method do?**
   - Answer

### 29. **Why do we have to zero the gradients?**
   - Answer

### 30. **What information do we have to pass to `Learner`?**
   - Answer

### 31. **Show Python or pseudocode for the basic steps of a training loop.**
   - Answer

### 32. **What is "ReLU"? Draw a plot of it for values from `-2` to `+2`.**
   - Answer

### 33. **What is an "activation function"?**
   - Answer

### 34. **What's the difference between `F.relu` and `nn.ReLU`?**
   - Answer

### 35. **The universal approximation theorem shows that any function can be approximated as closely as needed using just one nonlinearity. So why do we normally use more?**
   - Answer



## Further Research:

### 1. **Create your own implementation of `Learner` from scratch, based on the training loop shown in this chapter.**
> answer

### 2. **Complete all the steps in this chapter using the full MNIST datasets (that is, for all digits, not just 3s and 7s). This is a significant project and will take you quite a bit of time to complete! You'll need to do some of your own research to figure out how to overcome some obstacles you'll meet on the way.**
> answer


