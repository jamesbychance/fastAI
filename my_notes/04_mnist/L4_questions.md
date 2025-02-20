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
   - The pixel similarity approach works by first creating an "ideal" version of each digit through averaging all training examples. For classification, we take a new image and compare it pixel-by-pixel to each ideal digit. At each position, we calculate the difference between pixel values. These differences are summed across all pixels to measure total distance. The ideal digit with the smallest total distance is chosen as the classification. This method treats each image as a point in high-dimensional space where each dimension represents one pixel position.

### 4. **What is a list comprehension? Create one now that selects odd numbers from a list and doubles them.**
   - A list comprehension is a concise way to create a new list by applying an expression to each item in an existing list, often with a filter condition. It has the form: [expression for item in iterable if condition]
   - [number * 2 for number in my_list if number % 2 !=0]

### 5. **What is a "rank-3 tensor"?**
   - Answer

### 6. **What is the difference between tensor rank and shape? How do you get the rank from the shape?**
   - Answer

### 7. **What are RMSE and L1 norm?**
   - Answer

### 8. **How can you apply a calculation on thousands of numbers at once, many thousands of times faster than a Python loop?**
   - Answer

### 9. **Create a 3×3 tensor or array containing the numbers from 1 to 9. Double it. Select the bottom-right four numbers.**
   - Answer

### 10. **What is broadcasting?**
   - Answer

### 11. **Are metrics generally calculated using the training set, or the validation set? Why?**
   - Answer

### 12. **What is SGD?**
   - Answer

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

### 18. **Do you need to know how to calculate gradients yourself?**
   - Answer

### 18. **Why can't we use accuracy as a loss function?**
   - Answer

### 18. **Draw the sigmoid function. What is special about its shape?**
   - Answer

### 18. **What is the difference between a loss function and a metric?**
   - Answer

### 18. **What is the function to calculate new weights using a learning rate?**
   - Answer

### 18. **What does the `DataLoader` class do?**
   - Answer

### 18. **Write pseudocode showing the basic steps taken in each epoch for SGD.**
   - Answer

### 18. **Create a function that, if passed two arguments `[1,2,3,4]` and `'abcd'`, returns `[(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]`. What is special about that output data structure?**
   - Answer

### 18. **What are the "bias" parameters in a neural network? Why do we need them?**
   - Answer

### 18. **What does the `@` operator do in Python?**
   - Answer

### 18. **What does the `backward` method do?**
   - Answer

### 18. **Why do we have to zero the gradients?**
   - Answer

### 18. **What information do we have to pass to `Learner`?**
   - Answer

### 18. **Show Python or pseudocode for the basic steps of a training loop.**
   - Answer

### 18. **What is "ReLU"? Draw a plot of it for values from `-2` to `+2`.**
   - Answer

### 18. **What is an "activation function"?**
   - Answer

### 18. **What's the difference between `F.relu` and `nn.ReLU`?**
   - Answer

### 18. **The universal approximation theorem shows that any function can be approximated as closely as needed using just one nonlinearity. So why do we normally use more?**
   - Answer



## Further Research:

### 1. **Create your own implementation of `Learner` from scratch, based on the training loop shown in this chapter.**
> answer

### 2. **Complete all the steps in this chapter using the full MNIST datasets (that is, for all digits, not just 3s and 7s). This is a significant project and will take you quite a bit of time to complete! You'll need to do some of your own research to figure out how to overcome some obstacles you'll meet on the way.**
> answer


